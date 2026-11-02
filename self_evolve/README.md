# ViLaSR-self-evolve

把 ViLaSR 的 Stage-3 GRPO 换成 **self-evolve GRPO**：从 RRS 之后的 checkpoint
(`checkpoints/vilasr_reflective_4gpu/checkpoint-744`) 出发，用 VERSE 的
两轮自进化框架（Questioner → 确定性 Verifier → Solver）驱动训练，其中 Solver
使用 ViLaSR 的 **agentic think→draw→observe** 多轮画图 rollout。

```
Context pool (ScanNet 子集 + 合成 Maze 池)
  → Task Scheduler (按历史准确率自适应, 16 DGE 任务 + maze_navigation)
  → Round-1 Questioner (单轮, 出 ViLaSR 风格题)
  → Verifier: DGE (几何精确 GT) / MazeEngine (动作序列模拟 GT)
  → Round-2 Solver (agentic 多轮: <think> → Object Mapper / Path Tracer → <answer>)
  → GRPO 双轮更新 (r_Q: format+valid+obs;  r_A: 0.1*gcot_format + 0.9*GT_accuracy)
```

## 目录

| 路径 | 说明 |
|---|---|
| `easy_r1/` | 训练栈（VERSE easy_r1 fork + 本仓库改造） |
| `src/` | DGE (`simulator/tasks/rubrics/tools`) + `maze_engine.py`（新增确定性 Maze verifier） |
| `config/` | `api_secrets.env`（Ark VLM）/ `wandb.env` / `rubric_config.yaml` |
| `dataset/` | `maze_pool/`（300 张合成迷宫 + 结构 metadata）、`maze_contexts.jsonl` |
| `easy_r1/dataset/selfevolve_train.jsonl` | 训练上下文（ScanNet 子集 + maze，混洗） |
| `ScanNetData_local/` | 小型 ScanNet 子集（30 场景；frames/pose/depth/mesh/metadata） |
| `run_scripts/` | 数据准备 + 启动脚本 |
| `logs/` | 训练与数据准备日志 |

## 相对 VERSE 基座的核心改动

1. **Agentic Solver rollout**（`verl/workers/rollout/vllm_rollout_agentic.py`）
   移植自 ViLaSR `vllm_rollout_spmd_video.py`。Round-2 batch 带
   `meta_info["rollout_mode"]="agentic"` 时走多轮 think→draw→observe（最多
   `num_llm_calls_available` 次 LLM 调用，maze 额外 +10），工具调用画在图上回喂。
   返回 `multi_turn_mask`（仅 assistant token 参与 loss）、`multi_modal_inputs`
   （与最终多轮序列对齐的图像特征）、`agentic_sequence_text`（供 reward 做格式检查）。
2. **Actor loss mask**（`dp_actor.py`）：`response_mask &= multi_turn_mask`。
3. **fsdp_workers**：batch 已带 `multi_modal_inputs` 时跳过重算（防错位）。
4. **Round-2 提问 ViLaSR 化**（`ray_trainer.py`）：
   `_build_round2_agentic_question` 为方向/深度/可见性等任务生成 A)-D) 选项块
   （GT+干扰项 seeded shuffle），度量类任务加 numerical 后缀；
   `_build_answer_batch` 用 ViLaSR system prompt + 逐图 index caption 构造 Solver prompt。
5. **Maze 任务**（`src/maze_engine.py` + trainer 集成）：
   迷宫图池离线程序化生成（墙体/起点/A-D 终点 metadata 齐全）；Proposer 出
   "Full Action Sequence: Go up. Go left. ..." 风格题；引擎模拟序列
   （越界/撞墙 → invalid，落在 A-D → GT 字母）。注册为 `maze_navigation` 任务类型。
6. **Round-2 奖励**（`training/reward_function/agentic_solver_reward.py`）：
   gcot 多轮格式检查（每轮 `<think>`+Action、末轮 `<answer>`、工具 JSON 可解析）
   + 按任务类型的 GT 准确率（MC 字母映射匹配 / 数值 MRA）；invalid 题保留
   "解释失效原因" 分支（Ark LLM judge）。`unified_reward.py` 以
   `agentic_solver: true` 分发。
7. **Round-1 生成上限**：出题单轮 cap 1024 tokens（8192 预算属于 agentic Solver）。

## 数据准备（一次性）

```bash
# 1. 生成 maze 池（已完成: dataset/maze_pool, 300 张）
python run_scripts/gen_maze_pool.py

# 2. ScanNet 子集（30 场景, ~8GB, 场景清单见 ScanNetData_local/downloads/chosen_scenes.txt）
#    mesh:   aria2c（vh_clean_2.ply + segs + aggregation, TUM 官方服务器）
#    frames: run_scripts/fetch_25k_partial.py（HTTP Range 从 scannet_frames_25k.zip
#            只抽取所需场景的 color/pose/depth/intrinsics, 免下 5.6GB 整包）

# 3. 布局整理 + 训练 jsonl（ScanNet 上下文 + maze 上下文混洗）
python run_scripts/prep_scannet_subset.py --skip-extract --require-metadata

# 4. DGE metadata 提取（CPU-only）
bash run_scripts/run_metadata_extraction.sh
```

## 训练

```bash
# smoke（10 步全链路验证）
MAX_STEPS=10 SAVE_FREQ=-1 EXP_NAME=vilasr_selfevolve_smoke \
  bash run_scripts/run_selfevolve_4gpu.sh

# 正式验证运行（默认 200 步, wandb project: ViLaSR-SelfEvolve）
tmux new-session -d -s selfevolve "bash run_scripts/run_selfevolve_4gpu.sh"
```

- 只用 GPU 4-7（guardian-aware：只释放 4-7 + 占卡保活，见 `.cursor/rules/gpu-guardian.mdc`）。
- 配置：`easy_r1/examples/config_selfevolve.yaml`（TP=4、rollout n=4、batch 4、
  lr 1e-6、KL 0.01、prompt 6144 + agentic response 8192）。
- 监控：wandb `ViLaSR-SelfEvolve`；关键指标 `reward_round1/*`（出题质量）、
  `reward_round2/*`（accuracy / format / reflection / invalid_ratio）、
  `round1_scheduler/*`（自适应课程）。

## 已知事项

- 用户提供的 Ark key `REDACTED` 认证失败（"API key doesn't exist"，疑似
  Access Key ID）；当前使用此前已验证可用的 key（见 `config/api_secrets.env`）。
- ScanNet 完整数据（350GB）因 1TiB 空间限制不再保留，本仓库只维护 30 场景子集。
- 本机 flash-attn ViT kernel 损坏：启动脚本强制 sdpa / TORCH_SDPA（沿袭 VERSE）。
