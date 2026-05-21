# THUNLP OPD Code Reading Guide

这份笔记用于快速理解本项目的主 OPD 训练链路：从一个输入题目进入训练，到学生模型生成、教师模型打分、token-level OPD reward 计算，再到更新学生模型。

它关注的是主线 OPD 训练，不是完整复现论文里所有 SFT、GRPO、prompt selection、ablation 实验。

## 先知道这个项目在做什么

论文：Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe  
论文链接：https://arxiv.org/abs/2604.13016

核心问题：

- 学生模型自己生成回答，也就是 on-policy rollout。
- 教师模型不直接给最终分数，而是在学生走过的 token 状态上提供 token-level 分布信号。
- 训练目标是让学生在自己实际访问到的状态上，逐步向教师的高概率 token 分布对齐。

一个极简理解：

```text
数学题 prompt
  -> 学生模型生成解题过程
  -> 学生模型计算自己在每个 token 位置的 top-k logprob
  -> 教师模型在同一位置计算对应 token 的 logprob
  -> 比较 student 和 teacher 的 token 分布
  -> 得到 token-level distillation reward
  -> reward 作为 advantage
  -> 更新学生模型
```

## 推荐查看顺序

### 1. README：先看项目意图和参数

文件：

```text
README.md
```

重点看：

- `Training -> OPD`
- `Key Parameters`
- `SFT`
- `RL (GRPO)`

需要记住的几个概念：

- `ACTOR_MODEL_PATH`：学生模型，也就是被训练的 policy。
- `REWARD_MODEL_PATH`：教师模型，也就是提供 token-level reward 的模型。
- `ADV_ESTIMATOR=token_reward_direct`：OPD 主方法，token reward 直接变成 advantage。
- `LOG_PROB_TOP_K=16`：每个位置保留多少个 top-k token 来算分布差异。
- `TOP_K_STRATEGY=only_stu`：默认只取学生 top-k token，再查询教师对这些 token 的概率。
- `REWARD_WEIGHT_MODE=student_p`：reward 按学生概率加权。

### 2. 主入口脚本：看实验怎么启动

文件：

```text
on_policy_distillation.sh
```

重点看这些变量：

```bash
ADV_ESTIMATOR=token_reward_direct
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=7168
MAX_VAL_RESP_LENGTH=31744
MINI_BATCH_SIZE=64
N_RESPONSES=4
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
TRAIN_DATASET=datasets/dapo-math-17k.parquet
TEST_DATASET=["datasets/test_data/AIME25/test.parquet", "datasets/test_data/AMC23/test.parquet", "datasets/test_data/AIME24/test.parquet"]
```

最关键的一行是：

```bash
python3 -m verl.trainer.main_ppo
```

这说明真正训练入口在 `verl/verl/trainer/main_ppo.py`。

### 3. 简化复现脚本：先读这个更容易

文件：

```text
scripts/repro/minimal_opd_a1004.sh
```

这个脚本适合 smoke test 和理解主流程。它和 `on_policy_distillation.sh` 是同一个思路，但参数更少，适合先跑通。

建议你先对比：

```text
on_policy_distillation.sh
scripts/repro/minimal_opd_a1004.sh
```

看哪些参数是论文级训练用的，哪些参数是为了小规模测试降低成本。

### 4. PPO 入口：看 Ray 和 Trainer 怎么建起来

文件：

```text
verl/verl/trainer/main_ppo.py
```

重点看：

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)
```

这个文件做的事：

- 读取 Hydra 配置。
- 初始化 Ray。
- 创建 `TaskRunner`。
- 创建 `RayPPOTrainer`。
- 调用 trainer 的训练循环。

这里不是算法核心，只是训练系统入口。

### 5. 训练主循环：最重要的文件

文件：

```text
verl/verl/trainer/ppo/ray_trainer.py
```

直接看：

```python
class RayPPOTrainer
def fit(self)
```

这是整条训练链路的调度中心。建议按下面顺序读：

```text
fit()
  -> actor_rollout_wg.generate_sequences()
  -> actor_rollout_wg.compute_log_prob()
  -> rm_wg.compute_rm_score()
  -> actor_rollout_wg.compute_distillation_reward()
  -> compute_advantage()
  -> actor_rollout_wg.update_actor()
```

对应含义：

- `generate_sequences`：学生模型生成回答。
- `compute_log_prob`：学生模型计算生成 token 的 logprob 和 top-k logprob。
- `compute_rm_score`：教师模型计算对应 token 的 logprob。
- `compute_distillation_reward`：根据学生和教师的 logprob 差异计算 OPD reward。
- `compute_advantage`：把 reward 变成 advantage。
- `update_actor`：更新学生模型。

### 6. 学生模型 worker：看学生 top-k 和更新

文件：

```text
verl/verl/workers/actor/dp_actor.py
```

重点看三个函数：

```python
compute_log_prob()
compute_distillation_reward()
update_policy()
```

`compute_log_prob()` 做什么：

- 对学生生成的完整序列做 forward。
- 得到每个 response token 的 `old_log_probs`。
- 如果 `LOG_PROB_TOP_K > 0`，还会得到：
  - `student_top_k_ids`
  - `student_top_k_log_probs`

`compute_distillation_reward()` 是 OPD 核心。默认 `only_stu` 策略下，大致逻辑是：

```python
kl_val = student_logp - teacher_logp_on_student_tokens
rm_scores = -kl_val * weights
```

也就是：

```text
如果 teacher 比 student 更相信某个 token，这个 token 的 reward 更高。
如果 student 比 teacher 更相信某个 token，这个 token 的 reward 更低。
```

`update_policy()` 做什么：

- 取 `advantages`。
- 如果 advantage 是 3D，也就是 `(batch, seq_len, top_k)`，说明正在用 top-k OPD。
- 重新计算当前学生模型在这些 top-k token 上的 logprob。
- 用 policy gradient / PPO loss 更新学生。

### 7. FSDP worker：看 trainer 怎么调用模型 worker

文件：

```text
verl/verl/workers/fsdp_workers.py
```

重点看：

```python
class ActorRolloutRefWorker
generate_sequences()
compute_log_prob()
update_actor()

class RewardModelWorker
compute_rm_score()
```

这个文件是 Ray worker 层。它把 `ray_trainer.py` 里的调用，转发到真正的 actor、rollout、reward model 上。

其中 `RewardModelWorker.compute_rm_score()` 做教师模型 forward，产出：

- `teacher_on_student_log_probs`
- `teacher_top_k_ids`
- `teacher_top_k_log_probs`
- `teacher_entropy`
- `overlap_mask`

默认 `only_stu` 情况下，最重要的是 `teacher_on_student_log_probs`。

### 8. Advantage：看 reward 怎么变成训练信号

文件：

```text
verl/verl/trainer/ppo/core_algos.py
```

重点看：

```python
@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(...)
```

这个函数很简单：

```python
advantages = token_level_rewards * response_mask
returns = advantages.clone()
```

也就是说，本项目主 OPD 设置里，token-level reward 基本直接作为 advantage 使用。

## 用一个具体输入样本理解

假设训练集中有一条数学题：

```text
Question: Find the value of x if 2x + 3 = 11.
Answer:
```

训练过程大致是：

```text
1. DataLoader 读入 prompt。

2. 学生模型生成：
   "We solve 2x + 3 = 11. Subtract 3, 2x = 8. Therefore x = 4. \\boxed{4}"

3. 对学生生成的每个 token，学生模型计算：
   old_log_probs
   student_top_k_ids
   student_top_k_log_probs

4. 教师模型在同样的上下文位置上计算：
   teacher_on_student_log_probs

5. OPD reward 计算：
   reward(token) = -(log p_student(token) - log p_teacher(token)) * weight

6. token reward 变成 advantage：
   advantage(token) = reward(token)

7. update_actor 更新学生模型：
   让学生更倾向于教师认为更合理的 token 分布。
```

需要注意：OPD 不是简单模仿教师完整回答，而是在学生自己生成出来的轨迹上，让教师对每个 token 位置提供分布级指导。

## 先不要优先看的部分

这些文件或目录不是第一轮理解主 OPD 链路的重点：

```text
LlamaFactory/
grpo.sh
scripts/infer/vllm_rollout.py
scripts/val/eval/
verl/verl/trainer/config/
```

原因：

- `LlamaFactory/`：用于 SFT 冷启动，不是 OPD 主循环。
- `grpo.sh`：用于训练或复现教师 RL 模型，不是学生 OPD 主训练。
- `scripts/infer/vllm_rollout.py`：用于 teacher rollout + SFT 数据生成，不是 OPD token-level 训练。
- `scripts/val/eval/`：是评测生成和打分，先理解训练后再看。
- `config/`：配置很多，第一轮只需要知道脚本覆盖了哪些关键参数。

## 快速定位命令

在项目根目录下可以用：

```bash
rg -n "token_reward_direct|compute_distillation_reward|compute_rm_score|compute_log_prob|update_actor|generate_sequences" .
```

只查核心文件：

```bash
rg -n "compute_distillation_reward|compute_rm_score|compute_log_prob|update_policy" verl/verl/workers
rg -n "def fit|compute_advantage|generate_sequences|update_actor" verl/verl/trainer/ppo/ray_trainer.py
rg -n "token_reward_direct" verl/verl/trainer/ppo/core_algos.py
```

## 最短阅读路线

如果只想花 30 到 60 分钟快速理解实现，按这个顺序看：

```text
1. README.md
2. on_policy_distillation.sh
3. verl/verl/trainer/main_ppo.py
4. verl/verl/trainer/ppo/ray_trainer.py 的 fit()
5. verl/verl/workers/actor/dp_actor.py 的 compute_distillation_reward()
6. verl/verl/trainer/ppo/core_algos.py 的 compute_token_reward_direct_advantage()
7. verl/verl/workers/actor/dp_actor.py 的 update_policy()
```

读完这几个点，就能讲清楚这个项目的核心实现。
