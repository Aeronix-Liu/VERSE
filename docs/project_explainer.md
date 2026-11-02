# VERSE 项目详解

> **VERSE = Spatial Self-Evolve Vision-Language Agents（空间自进化视觉语言智能体）**
>
> 本文面向第一次接触本项目的人，目标是**从零看懂这个项目在做什么、为什么这样做、以及技术上怎么实现**。
> 全文分两大部分：先用通俗的话讲清楚「项目概览」，再逐层解剖技术细节
> （agentic 工具如何配置、SFT 数据、3D context 构建、self-evolve 全过程、过程奖励、消融实验、结果、代码地图）。
>
> 配套材料：[`../README.md`](../README.md)（发布版快速上手）。

---

# 第一部分 · 通俗概览

## 1. 我们要解决什么问题？

视觉语言模型（VLM，如 Qwen2.5-VL）能看图说话，但在**空间推理**上很弱——
让它判断「这个房间有几把椅子」「桌子离冰箱多远」「从起点怎么走到终点」这类
需要理解 3D 空间的问题时，常常答错。

原因是：模型只会「用文字在脑子里想」，而人类做空间推理时会**动手画**——
在图上圈出物体、画出路线、比对距离。**光靠文字思考，不够。**

## 2. 我们的核心思路（一句话）

> **让模型学会「边画边想」，并且「自己给自己出题、自己批改、自己进步」。**

拆成两个关键创新：

1. **Agentic 画图推理（think → draw → observe）**：给模型配两支"画笔"工具——
   - **Object Mapper**：在图上画一个物体的**边界框（bounding box）**；
   - **Path Tracer**：在图上画一条**移动路径**。

   模型的推理变成一个多轮循环：先「思考」，再「画一笔」，系统把画好的图**回喂**给模型，
   模型「观察」自己画的对不对、再调整……直到给出最终答案。推理过程本身就是"画图"。

2. **Self-Evolve（自进化 GRPO）**：训练时**不依赖任何人工标注的强化学习数据**。
   模型自己在 3D 场景里**出题**（Proposer），一个**确定性引擎**（不是另一个大模型，而是一套
   基于几何计算的程序）来**验证题目是否合法并算出标准答案**，然后模型再**自己答题**（Solver），
   答对了就强化。相当于模型自建了一套"习题册 + 标准答案"，不断自我出题、自我训练。

## 3. 为什么这件事有意思？

- **不需要人工标注 RL 数据（label-free）**：训练信号来自一个能**精确计算**标准答案的引擎
  （ScanNet 三维几何 / 迷宫 BFS 最短路），而不是人工标签。模型等于自建课程（curriculum）。
- **不仅能验证"答案"，还能验证"过程"**：因为工具画出来的是**几何对象**（框、路径），
  而引擎知道**真实几何**，所以我们能检查"模型画的框是不是真的框在题目问的物体上"、
  "画的路径是不是真的沿着最短路走"——这是一种**确定性的过程奖励（process reward）**，
  纯粹只看最终答案的 RL 是拿不到这种信号的。它专治"随便画一个框、然后蒙答案"的脱节现象。
- **数据高效**：最好的 self-evolve run 用**一半的训练步数**就达到了之前的最好 VSI 分数，并提升了迷宫任务。

## 4. 目标基准（benchmark）

- **VSI-Bench**：视频空间智能（多帧视频，问计数/距离/方向/大小/房间尺寸/路线规划等）。
- **SPAR-Bench**：单图/多视角空间关系（深度、距离、相对关系、位置匹配、视角变化等）。
- **MAZE**：网格迷宫导航（给一张迷宫图，从起点走到 A/B/C/D 某个终点，选哪条动作序列）。

基座模型自始至终是 **Qwen2.5-VL-7B-Instruct**。

## 5. 三阶段流水线（鸟瞰）

```
Qwen2.5-VL-7B-Instruct（基座）
      │
      │  Stage 1: Cold-Start SFT（学会画图工具的"语法"）
      ▼
  cold-start 模型 (cs-3258)
      │
      │  Stage 2: Reflective Rejection Sampling SFT（学会"画→看→改"的反思式推理）
      ▼
  RRS 模型 (rrs-744)  ←——— 这是 self-evolve 的初始策略 π_ref（KL 锚点）
      │
      │  Stage 3: Self-Evolve GRPO（自己出题→引擎验证→自己答题→强化）
      ▼
  self-evolve 模型（本项目的核心贡献）
```

前两个阶段是**监督微调（SFT）**，教会模型基本的画图工具用法和反思习惯；
第三阶段是**强化学习（GRPO）**，也就是 self-evolve 的主战场。

---

# 第二部分 · 技术细节解剖

## 6. Agentic 工具是如何配置给模型的

**关键点：这两个工具不是 OpenAI 那种 function-calling API，而是"提示词定义的伪工具"。**
模型被 system prompt 要求输出**特定格式的文本**（`Action:` + 一段 ```json``` 代码块），
运行时用程序解析这段文本、用 PIL 在图上画出来、再把画好的图作为新的一轮输入喂回去。

### 6.1 System Prompt（工具定义）

系统提示词在多处逐字一致，权威版本见
`ViLaSR/eval/infer.py` L23–80、`ViLaSR-self-evolve/easy_r1/verl/utils/agentic.py` L29–86。核心内容：

```
### Available tools:

1. Object Mapper
- Purpose: Identifies and maps key items in the space
- Input format: JSON
  [{ "index": i, "bbox_2d": [x1, y1, x2, y2], "label": "object name" }]
- Output: Generates bounding boxes for visual inspection of the i-th image

2. Path Tracer
- Purpose: Plots movement or connections between points
- Input format: JSON
  [{ "index": i, "start_point_2d": [x1,y1], "end_point_2d": [x2,y2], "label": "trace_description" }]
- Output: Generates visual paths for verification of the i-th image

### Required Output Format:
<think> [推理过程] </think> Action: [Object Mapper/Path Tracer]
```json
[坐标 JSON]
```

最终回答：
<think> [推理过程] </think> Action: Answer
<answer> [最终答案] </answer>
```

坐标是**绝对像素**，左上角 `(0,0)`、右下角 `(width, height)`（system prompt 里用运行时真实宽高填充）。

### 6.2 每一轮的 JSON 动作格式

| 工具 | 必需字段 | 说明 |
|---|---|---|
| **Object Mapper** | `bbox_2d`（4 个整数）、`label` | `index`：1-based 图像序号，选中要画在第几张图上 |
| **Path Tracer** | `start_point_2d`、`end_point_2d`、`label` | `index`：同上 |

### 6.3 解析 → 绘制 → 回喂 的闭环机制

1. **解析**：`edit_image_video.py::parse_bbox_and_movement` 从模型输出里抠出 ```json``` 块，
   用 `ast.literal_eval` 解析，按字段区分是 bbox 还是 movement。
2. **绘制**：`plot_bounding_boxes`（彩色矩形 + label 文本）/ `plot_movement`
   （黑线 + 红色起点/蓝色终点圆点）把标注画到**源图的副本**上（按 `index` 选源图）。
3. **回喂**：把画好的标注图 append 到 `multi_modal_data['image']`，并在对话序列里追加一段新的
   user 轮次，内含 Qwen-VL 的视觉占位符 `<|vision_start|><|image_pad|><|vision_end|>` +
   "Continue your reasoning…"。下一次 vLLM 生成时，模型就"看到"了自己刚画的图。

### 6.4 多轮循环的收尾

- **轮数预算** `num_llm_calls_available`：eval 默认 10 轮（迷宫 20 轮）；ViLaSR GRPO 训练 10 轮；
  **self-evolve 训练 6 轮**（迷宫样本额外 +10 轮）。
- **结束条件**：模型输出完整的 `<answer>...</answer>`（自然停止）、或达到长度上限、或工具解析失败、
  或图片数量超上限。轮数耗尽前的最后一轮会用 `final_prompt` 告知模型"不能再画了，请直接回答"。
- 日志里偶尔出现的 `parse_bbox_and_movement TypeError`（模型画出畸形 JSON）是**被捕获的良性异常**，
  不影响训练。

一图流程：

```
[System: 工具定义]  [User: 图 + 问题]
   → 生成: <think>…</think> Action: Object Mapper ```json[...]```
   → 解析+绘制到图上 → 新增标注图 + "Continue…" 用户轮
   → (重复直到 <answer> 或轮数耗尽)
   → 生成: Action: Answer <answer>…</answer>  → 完成
```

---

## 7. SFT 数据（Stage 1 & 2）

两个 SFT 阶段都直接**复用 ViLaSR 官方释放的数据**（本项目做了图片清洗），格式均为 **ShareGPT JSON**
（`conversations` + `images`），配置在 `ViLaSR/ViLaSR-data/dataset_info.json` / `VERSE/sft/dataset_info.json`。

### 7.1 Cold-Start SFT（Stage 1）— 教"画图语法"

- **数据来源（4 个）**：`SR_91k`（11,352）、`VQA`（10,000）、`maze`（7,298）、`GPT4Scene`（6,095），
  清洗后共约 **34,745** 条（上游名 "ViLaSR-ColdStart-33k"）。
- **一条轨迹长什么样**：`system`（工具语法）→ `user`（`<image>` + 问题 + "Begin your reasoning…"）
  → `assistant`（`<think>…</think> Action: Object Mapper/Path Tracer` + JSON）→ `user`（回喂标注图 + "Continue…"）
  → …循环…→ `Action: Answer` + `<answer>`。
- **超参**（`VERSE/sft/cold_start.yaml`）：全量 SFT，**冻结视觉塔**（freeze_vision_tower=true），
  DeepSpeed ZeRO-3，有效 batch = 4 GPU × 1 × 8 = 32，lr 1e-5，3 epochs，cutoff 18432。
  → 34,745 样本 ≈ **3,258 步**（3 epoch），产出 `cs-3258`。

### 7.2 Reflective Rejection Sampling SFT（Stage 2）— 教"反思式画图"

- **数据**：同 4 个来源的**反思轨迹**，清洗后约 **7,912** 条（上游名 "ViLaSR-RSS-8k"），
  轨迹是**多轮 think→draw→observe→revise**（比 cold-start 更长，平均 8~24 张图、12~18 轮）。
- **"反思"的定义**：同一个 `label` 在轨迹中被用**不同的 `bbox_2d` 坐标重画**——即模型观察了上一步的
  标注反馈后**修正**了自己的框（见 `gcot.py` L104–133 的 reflection 判据）。上游只保留 `score=1.0`
  （答对 ∧ 格式合格 ∧ 含反思）的轨迹做 SFT。
- **超参**：从 `cs-3258` 继续 SFT，其余同 Stage 1 → 7,912 样本 ≈ **744 步**，产出 `rrs-744`。
  **`rrs-744` 是所有 self-evolve run 的初始策略 π_ref（KL 锚点）。**

> 经验结论：**SFT 两阶段贡献了大部分涨点**（VSI: 基座 34.2 → cold-start 39.2 → RRS 39.6），
> Stage-3 RL 在此基础上做温和但真实的增量。迷宫能力（MAZE 90+）也主要来自 RRS 阶段。

---

## 8. 3D Context 如何构建（self-evolve 的"素材库"）

self-evolve 训练不喂现成的题，只喂 **3D context（场景素材）**，题目是训练中现场生成的。
每行 JSONL 只有图片路径：`{"problem": null, "answer": null, "images": [...]}`。

### 8.1 ScanNet 逐帧元数据（GT 的来源）

素材来自 **ScanNet** 真实室内扫描（`ScanNetData_local/`，332 个装备齐全的场景）。关键是**离线预计算的元数据**：

- **场景级**（`metadata/{scene}/{scene}.json`）：`room_size`（房间面积 m²）、每个物体的
  `object_id / label / size（最长边 cm）/ 3d_bbox（轴对齐包围盒，米）/ 3d_location（质心）/ first_frame`。
- **逐帧级**（`metadata/{scene}/frame_processed/{frame}.json`）：每帧可见物体的
  **`2d_bbox`（像素）**、`visibility`（0–1）、`camera_location`、以及从场景级拷贝的 3D 信息。

**2D bbox 怎么来的**（`bbox_extractor.export_2d_bbox`）：加载 mesh 顶点 → 用 `{scene}.txt` 的
**轴对齐矩阵**校正 → 用相机**内参 + 位姿**把每个物体实例的点云**投影**到像面（带深度遮挡检测）
→ 取可见投影点的轴对齐 2D 外包 ± 2px。**这些精确的 2D/3D 几何就是引擎算标准答案、以及后续验证
"画的框对不对"的依据。**

### 8.2 四种模态的 context（v3 构建器）

`build_selfevolve_train_v3.py` 按图片数量/路径区分模态：

| 模态 | 怎么构建 | 对应任务 |
|---|---|---|
| **scene**（≥3 帧） | 每场景 3 组 × 8 帧的**覆盖率贪心子集** | VSI 系（计数/距离/大小/方向/路线/房间） |
| **single_image**（1 帧） | 每场景 ≤7 个均匀视角帧 | SPAR 单图任务 |
| **image_pair**（2 帧） | 每场景 3 对宽基线帧对 | SPAR 多视角任务 |
| **maze** | 迷宫池全量 750 张 | maze_navigation |

> 当前主力训练用的是 **v4**（无 SPAR：只保留 scene 822 行 + maze 750 行），因为消融显示加 SPAR 并未涨点。

**覆盖率贪心选帧（coverage-greedy）**：为了让"问的物体在选中的帧里真的看得见"，
从场景所有帧里**贪心地挑 k 帧，使可见物体（`object_id`）的并集最大**；第一帧强制用不同的高覆盖帧作起点，
从而得到 3 组不同的子集。这样能显著降低"问了看不见的物体"带来的 reward 噪声。

### 8.3 迷宫池（maze pool）

- `gen_maze_pool.py`：4×4/5×5/6×6 各 100 个 → 300 个；`extend_maze_pool.py` 再加 450 个 → **共 750**。
- 生成算法（`maze_engine.py`）：递归回溯生成完美迷宫 → 敲掉 ~15% 额外墙 → 随机起点 + 贪心放置 A/B/C/D 四个终点
  （彼此曼哈顿距离 ≥2）→ **BFS 算每个终点的最短路** → 渲染 616×616 PNG（28 对齐，绿点起点、红字 A/B/C/D）。
- 元数据存 `start / destinations / passages / shortest_paths(letter→动作序列)`——**BFS 最短路即标准答案**。

---

## 9. Self-Evolve 详细过程（Stage-3 GRPO 核心）

每个训练 step 是一个**两轮循环**（`ray_trainer.py` 编排）。`batch=16`（16 个 context），`rollout.n=8`
（每题 8 条 rollout），**一个 step 只做一种 task type**（由 schedule 决定，全 batch 同类型）。

### 9.1 Round 1 — Proposer（出题）

- 系统按 schedule 取本步 task type，从对应模态的 dataloader 取 16 个 context。
- **structured_imitate 模式**：Proposer 不写自由文本，只输出 `<question_params>`——即从场景物体池里
  **选标签**（例如选 "chair" 做计数、选 "table, refrigerator" 做距离）。迷宫用 **round_robin** 模式：
  系统按 A/B/C/D 轮转指派终点（Proposer 输出被忽略），保证 4 个不同且合法的题、杜绝多样性坍塌。
- **系统渲染题面**：用固定模板把选中的参数渲染成正式问题文本（且**逐字对齐 benchmark 风格**，
  见 §9.3）。
- **确定性引擎验证 + 算标准答案**（下节）。

### 9.2 DGE — 确定性标准答案引擎（Task → Rubric → Tool）

ScanNet 场景任务的验证与答案计算走 **DGE（Deterministic Ground-truth Engine）**，三层结构（`src/`）：

- **Task**（`tasks/`）：定义如何从问题里提取参数、需要哪些工具。
- **Rubric**（`rubrics/`）：定义工具调用的**步骤序列**。
- **Tool**（`tools/`）：纯几何计算（基于 §8.1 的 ScanNet 元数据）。

两个具体例子：

- **object_counting**：Rubric = `vlm_tool`(抽取类别) → `ambiguity_checker`(该 label 必须唯一) →
  `object_count_tool`(数场景里匹配该 label 的物体个数)。GT = `count`。
- **absolute_distance**：Rubric = 抽取两个物体 label → 校验各自唯一 → `absolute_distance_tool`
  计算两个物体 **3D 轴对齐包围盒的最近点欧氏距离**。GT = `"{d} meters"`。

迷宫任务则由 **MazeEngine**（BFS 最短路）验证与给答案。
验证不通过的题（如选的物体不在场景里/帧中不可见）判为 **invalid**——这是给 Proposer 的真实训练信号。

### 9.3 Round-2 题面的 benchmark 对齐

原则：**Round-1 文本只服务于 DGE 验证，可以简单；Round-2 文本是 Solver 的训练输入，必须和 bench 逐字对齐**
（含选项字母 `A. `、`relative_distance` 恒 4 选项、`relative_direction_hard` 按 GT 分流成 bench 原句、
counting 用 bench 唯一句式、SPAR 的 marker 引用与 canonical 句式重建等）。这样 solver 学到的分布才能迁移到 eval。

### 9.4 Round 2 — Solver（答题）

- 对每个 **valid** 的题，用 §6 的 **agentic think→draw→observe** 多轮 rollout 生成 8 条答案路径
  （默认 6 次 LLM 调用，迷宫 +10）。
- invalid 的题走"解释为何无效"分支（由一个 LLM judge 打小分）。

### 9.5 奖励

- **Proposer 奖励** `r^Q`：设计上是 `0.1·格式 + 0.9·(validity · observation质量)`，但在 structured_imitate 下
  observation 未被解析（已知的代码-设计缺口），实际退化为 `0.1·格式分`——Proposer 主要从"参数可解析/清洗通过"获得梯度。
- **Solver 奖励** `r^A`（**这是我们真正塑形的奖励**）：
  ```
  r = 0.1·format + (0.9 - w)·accuracy + w·Φ
  ```
  - `accuracy`：MC 用字母匹配、数值用 MRA（Mean Relative Accuracy）；对照 **引擎给的标准答案**。
  - `Φ`：**引擎验证的过程奖励**（§10），由权重 `w` 门控；`w=0` 时纯 outcome（`0.1·fmt + 0.9·acc`）。

### 9.6 GRPO 目标（双轮）

- Proposer 组与 Solver 组**分别做组内归一化**优势 `Â=(r-μ)/σ`。
- 损失为 **DAPO 式非对称裁剪 + dual-clip**（clip 区间 `[0.8, 1.3]`，dual-clip 常数 3.0），
  **low-variance KL 以 loss 项形式锚定 `rrs-744`**（β=0.01，防止"自出题-自训练"闭环漂移）。
- `multi_turn_mask` 保证只训练 assistant token（工具回喂的观测图 token 不计入梯度）。
- 超参：lr 1e-6，TP=2，max_len 32768。

### 9.7 调度器

`SCHEDULER=round_robin` 读一个 schedule JSON（如 `full_selfevolve_schedule_v3.json`，200 步，
迷宫 50 + 各 VSI 任务 ~21 步），每步固定一种 task type；dataset 预排序 + `shuffle:false` 保证每步 batch 模态同质。

---

## 10. 过程奖励 Φ（本项目的差异化创新）

纯 outcome RL 只看最终答案对不对，看不到"模型是不是随便画个框然后蒙对了"。
因为**推理即画图**、且引擎知道真实几何，我们能确定性地验证**画图过程**本身：

- **迷宫路径保真度**（`maze_trajectory_fidelity`）：把 Path Tracer 画的每段端点吸附到最近网格格，
  得到"画出的格路径"，与题目动作序列展开的"期望格路径"求
  `Φ = 0.6·LCS(画,期望)/|期望| + 0.4·终点命中`。
- **VSI 物体接地度**（`vsi_grounding_fidelity`）：把题目相关物体的 GT 2D bbox 缩放到 Solver 作画的
  processed 坐标系，与模型 Object-Mapper 画的同 label 框求 **max IoU**，对相关标签取平均。
  `Φ = mean_over_labels( max IoU )`，并记录 hit-rate / coverage 等诊断。

由环境变量 `SPATIALEVO_MAZE_FIDELITY_WEIGHT` / `SPATIALEVO_VSI_GROUNDING_WEIGHT` 门控（`w`）：
`w=0` 时只计算不塑形（纯诊断）；最好的 run 用 `w=0.2`（accuracy 仍以 0.7 主导）。
这是一种**确定性、零额外模型、零标注的 PRM 式过程奖励**——只有在"推理=画图 + 确定性引擎"的设定下才可能。

---

## 11. 消融实验（Ablation Study）

> **原则：所有 ablation 代码独立存放在 `ViLaSR-self-evolve/ablations/<名字>/`，绝不改动正常实验代码。**
> 每个 ablation 都从 `rrs-744` 起训、100 步、GPU 0-7、其余 recipe 与 grounding run 完全一致，只改动一个变量。

### 11.1 Ablation #1 — self-evolve without GT（用 majority voting 代替标准答案）

- **位置**：`ablations/no_gt_majority_vote/`。
- **改动**：丢弃引擎给的**标准答案**，改用**自洽性投票（TTRL 式）**——在每个 GRPO 组（同一题的 8 条 rollout）内
  取多数答案作为伪标签（MC 取字母众数、计数取整数众数、连续值取中位数），按与多数的一致性给奖励。
  引擎仍用于**验证题目合法性 + 渲染题面**，仅去掉 GT 答案。
- **诊断**：额外记录 `majority_vote/consensus_ratio`（组内一致率）与 `pseudo_vs_true_acc`（伪标签 vs 真 GT 的吻合度）。
- **实现**：`reward_function/agentic_solver_majority_reward.py`（复用原始 reward 的全部辅助函数，仅替换 valid 样本的
  accuracy 计算）+ `unified_reward_majority.py`（Round-1 不变，Round-2 换成多数投票）；16 项 CPU 单测。

### 11.2 Ablation #2 — no process reward（仅 outcome reward）

- **位置**：`ablations/no_process_reward/`。
- **改动**：把过程奖励权重强制为 0（`SPATIALEVO_VSI_GROUNDING_WEIGHT=0`、`SPATIALEVO_MAZE_FIDELITY_WEIGHT=0`），
  Round-2 奖励退化为纯 outcome `0.1·format + 0.9·accuracy`。**保留 GT**（与 #1 正交）。
- **实现**：一个薄 wrapper，设好环境变量后 `exec` 未改动的生产 launcher `run_phase2_imitate_grounding.sh`——
  因为"无过程奖励"本质就是现有 reward 在 `w=0` 时的行为，无需改任何 reward 代码。
- **定位**：它是 grounding run（`w=0.2`）的**直接对照**（`w=0.2` vs `w=0`），用来隔离过程奖励的贡献。

---

## 12. 结果（官方口径；VSI 8-类均值 / SPAR 20-型 / MAZE 准确率）

| 模型 / 检查点 | 协议 | VSI | SPAR | MAZE | 备注 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B 基座 | direct | 34.21 | 34.12 | 33.5 | 无工具 |
| + Cold-start SFT (cs-3258) | agentic | 39.2 | — | — | 工具语法 |
| + RRS SFT (**rrs-744**) | agentic | 39.58 | 35.17 | 90.35 | RL 初始策略 |
| static GRPO-200（外部数据） | agentic | 42.1 | 35.4 | 91.5 | 上界参考 |
| self-evolve 0811-200（outcome, 有 GT） | agentic | 40.52 | 35.66 | 89.15 | 最佳纯 self-evolve |
| grounding s50（过程奖励 w=0.2） | agentic | 40.53 | 34.78 | 89.65 | **半步数达到 40.5** |
| grounding s100（过程奖励 w=0.2） | agentic | 40.52 | 35.04 | 90.10 | MAZE↑ |
| **Ablation#1 no-GT majority s100** | agentic | 40.20 | 34.75 | 89.55 | 多数投票≈GT |
| **Ablation#2 no-process s100** | agentic | *训练收尾，待评测* | | | w=0 对照 |
| ViLaSR-7B（官方全量模型） | agentic | 45.15 | 37.32 | 93.70 | 含官方完整 Stage-3 RL |
| SpatialEvo-7B | direct | 40.60 | 34.94 | 32.80 | 无工具，迷宫弱 |
| SpatialThinker-7B | direct+模板 | 32.42 | 33.36 | 33.00 | 单图模型，VSI 视频不匹配 |

**要点**：
- **Ablation#1（no-GT 多数投票）在 VSI/MAZE 上基本追平有 GT 的 self-evolve**（40.20/89.55 vs 40.52/89.15）——
  说明在此设定下自洽性可近似替代引擎 GT，是论文的一个有意思的发现。
- grounding 过程奖励用**一半步数**（50 步）就达到 VSI 40.5，并提升 MAZE——过程信号提升了数据效率。
- 三个外部基线各用其**原生协议**评测（ViLaSR-7B agentic、SpatialEvo/SpatialThinker direct，
  SpatialThinker 加其官方模板），因此可比。ViLaSR-7B 分数高是因为它是含完整 Stage-3 RL 的官方全量模型，
  而我们的 `rrs-744` 只到 Stage-2。

---

## 13. 代码地图 & 复现要点

### 13.1 关键路径

| 用途 | 路径 |
|---|---|
| 两轮循环编排 | `ViLaSR-self-evolve/easy_r1/verl/trainer/ray_trainer.py` |
| Round-2 Solver 奖励 | `.../easy_r1/training/reward_function/agentic_solver_reward.py` |
| Round-1 Proposer 奖励 | `.../question_quality_reward.py`；统一分发 `unified_reward.py` |
| agentic 工具/system prompt | `ViLaSR/eval/infer.py`、`.../easy_r1/verl/utils/agentic.py`；绘制 `edit_image_video.py` |
| DGE 引擎 | `ViLaSR-self-evolve/src/`（`simulator/world_simulator.py` + `tasks/`+`rubrics/`+`tools/`） |
| 迷宫引擎 | `ViLaSR-self-evolve/src/maze_engine.py` |
| context 构建 | `run_scripts/build_selfevolve_train_v4.py`；ScanNet 元数据 `metadata_extractor_small_mesh.py` |
| SFT 配置 | `ViLaSR/train/{cold_start,reflective_rejection_sampling}/...`、`VERSE/sft/` |
| 评测 | `ViLaSR/eval/infer.py`（agentic）/ `infer_direct_vllm.py`（direct）/ `evaluate.py`（打分） |
| 消融 | `ViLaSR-self-evolve/ablations/{no_gt_majority_vote,no_process_reward}/` |

### 13.2 环境与硬件

- 硬件：8× A800 (80GB)。SFT 用 DeepSpeed ZeRO-3；self-evolve 用 FSDP + vLLM rollout（TP=2）。
- Conda：训练 `Xinran_SpatialEvo`；评测/推理 `xinran_verl_demo`（VERSE 里分 `verse-sft`/`verse`/`verse-eval`）。
- 中国镜像：`HF_ENDPOINT=https://hf-mirror.com`；GitHub 走 `https://gh-proxy.com/`。
- **GPU Guardian（硬约束）**：这是共享节点，任何 GPU 长时间空闲会导致整个节点被回收。训练/评测脚本都
  只 release 自己要用的卡、跑完让 guardian 自动重占，**绝不 `--stop` 全部**。

### 13.3 典型运行

```bash
# self-evolve 训练（示例：加过程奖励，200 步）
cd ViLaSR-self-evolve
SPATIALEVO_VSI_GROUNDING_WEIGHT=0.2 EXP_NAME=... MAX_STEPS=200 \
  bash run_scripts/run_phase2_imitate_grounding.sh

# 消融 #1（no-GT 多数投票，100 步，8 卡）
bash ablations/no_gt_majority_vote/run_ablation_no_gt_majority.sh
# 消融 #2（no process reward，100 步，8 卡）
bash ablations/no_process_reward/run_ablation_no_process.sh

# 评测（FSDP checkpoint 先 merge，再三 bench agentic 评测 + 官方口径打分）
bash ablations/no_gt_majority_vote/eval_nogt_majority.sh   # 可克隆改 EXP 名
```

---

## 14. 踩过的坑（精选）

- **迷宫多样性坍塌**：`pick_letter` 模式下 Proposer 收敛到同一字母 → 改 `round_robin` 系统轮转指派。
- **SPAR reward 噪声**：spatial_relation 的"密码字母表"、depth 返回列表字符串等 3 个 bug 导致 reward≈随机，
  已重写为 bench 三元组/数值并对齐题面（但后续消融显示加 SPAR 未涨点，主力 recipe 去掉了 SPAR）。
- **系统时钟是 UTC、用户是 UTC+8**：判断"训练是否卡住/耗时"必须先换算，曾因此误杀健康训练。
- **`parse_bbox_and_movement` 的 Traceback 是良性**：模型偶发畸形工具调用被 try/except 捕获，不影响训练。
- **VSI 官方口径**：必须用 **8 类别均值**（而非样本加权），否则会高估约 2 个点。

---

*本文由项目当前代码与实验记录整理；如需更细的数学形式（三阶段训练目标、双轮 GRPO 目标函数、
过程奖励公式），见 `cursor_memory/2026-08-08_ViLaSR-self-evolve-architecture-explained.md` §8 与
`cursor_memory/2026-08-23_project-summary-and-method.md`。*
