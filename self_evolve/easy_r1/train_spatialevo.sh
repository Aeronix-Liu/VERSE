set -x

# 4x A800. Stop local VLM on :8810 first if it occupies GPU 1 (bash Xinran/scripts/stop_local_qwen3vl.sh).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWANLAB_MODE=cloud

MODEL_PATH="${SPATIALEVO_POLICY_MODEL_PATH:-/data/pengtianshuo/self_evolve/models/Qwen3-VL-8B-Instruct}"
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=1

python3 -m verl.trainer.main \
    config=examples/config_spatialevo.yaml \
    data.train_files=./dataset/spatialevo_train.jsonl \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.project_name=VERSE \
    trainer.experiment_name=VERSE_qwen3vl8b \
    trainer.save_checkpoint_path=./checkpoints/spatialevo_qwen3vl8b \
    trainer.logger='["file"]' \
    trainer.n_gpus_per_node=4
