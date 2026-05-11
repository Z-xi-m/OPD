#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
BASE_DIR=${BASE_DIR:-/data/zhu.ximo/opd_repro}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export HF_HOME=${HF_HOME:-$BASE_DIR/hf_home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$BASE_DIR/cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$BASE_DIR/vllm_cache}
export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-$BASE_DIR/cache/outlines/$(uuidgen)}
export TMPDIR=${TMPDIR:-$BASE_DIR/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-$BASE_DIR/ray_tmp}

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$OUTLINES_CACHE_DIR" "$TMPDIR" "$RAY_TMPDIR"

export PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillationSmoke}
export ADV_ESTIMATOR=token_reward_direct
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-4}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-1}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-smoke}
export TEST_DATASET=${TEST_FILE:-'["datasets/test_data/AIME24/test.parquet"]'}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$BASE_DIR/models/Qwen3-1.7B-SFT}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$BASE_DIR/models/Qwen3-4B-Base-GRPO}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=${PROJECT_PATH:-$BASE_DIR/outputs/thunlp_opd}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-topk_${LOG_PROB_TOP_K}-$(date +%Y-%m-%d_%H-%M-%S)}
export CKPT_PATH=${CKPT_PATH:-$PROJECT_PATH/checkpoints/$EXPERIMENT_NAME}

mkdir -p "$PROJECT_PATH/logs" "$PROJECT_PATH/validation_log" "$PROJECT_PATH/checkpoints"

LOG_FILE="$PROJECT_PATH/logs/${EXPERIMENT_NAME}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/verl:${PYTHONPATH:-}"

ray stop --force || true
ray start --head --temp-dir="$RAY_TMPDIR"
sleep 5

KL_ARGS=()
if [ "$USE_KL" = "True" ]; then
  KL_ARGS=(
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.005
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
  )
else
  KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 8192) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 8192 ))

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.grpo_outcome_weight="$GRPO_OUTCOME_WEIGHT" \
  data.shuffle=False \
  data.train_files="$TRAIN_DATASET" \
  data.val_files="$TEST_DATASET" \
  data.train_batch_size="$((MINI_BATCH_SIZE * PARALLEL_SIZE))" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESP_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="$PARALLEL_SIZE" \
  "${KL_ARGS[@]}" \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
  +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
  +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
  +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$PARALLEL_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.n="$N_RESPONSES" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.val_kwargs.max_tokens="$MAX_VAL_RESP_LENGTH" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=True \
  +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
  reward_model.model.path="$REWARD_MODEL_PATH" \
  reward_model.model.input_tokenizer=null \
  reward_model.model.use_remove_padding=True \
  reward_model.model.fsdp_config.param_offload=False \
  +reward_model.model.dtype="$MODEL_DTYPE" \
  reward_model.micro_batch_size_per_gpu=1 \
  custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
  custom_reward_function.name=reward_func \
  trainer.val_before_train=False \
  trainer.log_val_generations=1 \
  trainer.logger="['console']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.validation_data_dir="$PROJECT_PATH/validation_log/$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  trainer.default_local_dir="$CKPT_PATH" \
  trainer.is_plot=False
