"""
确定性评估脚本 —— 判断策略真实成功率的依据。

为什么不直接看 TensorBoard：
  训练曲线里的 "Total reward" 是回合累加值，回合越长数字越大，
  两个策略之间没法直接比。而且曲线回答不了"成功率到底是多少"。

口径设计（两个关键决定）：
  1. 确定性动作 —— 用策略输出的均值，不采样，消除随机性。
  2. 只统计每个环境的【第一个回合】—— 成功的回合结束得更快、更快重开，
     若统计累计次数，跑得快的环境会被重复计入分子，成功率会算高。

用法:
    uv run python eval_policy.py <policy_path> [num_envs] [max_steps]
"""

import sys

import numpy as np
import torch
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl.skrl.torch import wrap_env
from motrix_rl.skrl.torch.train.ppo import Trainer, _get_cfg

ENV_NAME = "vbot_navigation_section01"
POLICY_PATH = sys.argv[1]
NUM_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

trainer = Trainer(ENV_NAME, None, enable_render=False,
                  cfg_override={"play_num_envs": NUM_ENVS, "seed": 42})
rlcfg = trainer._rlcfg

env = env_registry.make(ENV_NAME, sim_backend=None, num_envs=NUM_ENVS)
set_seed(rlcfg.seed)
skrl_env = wrap_env(env, False)
models = trainer._make_model(skrl_env, rlcfg)
agent = trainer._make_agent(models, skrl_env, _get_cfg(rlcfg, skrl_env))
agent.load(POLICY_PATH)

print(f"policy   : {POLICY_PATH}")
print(f"num_envs : {NUM_ENVS}   max_steps: {MAX_STEPS}")

first_done = np.zeros(NUM_ENVS, dtype=bool)
first_success = np.zeros(NUM_ENVS, dtype=bool)
first_fail = np.zeros(NUM_ENVS, dtype=bool)
steps_to_done = np.full(NUM_ENVS, -1, dtype=np.int64)

with torch.no_grad():
    obs, info = skrl_env.reset()
    for step in range(1, MAX_STEPS + 1):
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, reward, terminated, truncated, info = skrl_env.step(actions)

        # 结局必须从环境对象上读：info 里的标志会在回合结束后被
        # _reset_done_envs 清掉，step() 返回时已经读不到了。
        outcome = env.last_episode_outcome          # 0=未结束 1=成功 2=失败
        ending = (env.episodes_completed_per_env > 0) & ~first_done
        if ending.any():
            first_success |= ending & (outcome == 1)
            first_fail |= ending & (outcome == 2)
            steps_to_done[ending] = step
            first_done |= ending

        if step % 1000 == 0 or first_done.all():
            print(f"  step {step:>4}: 已结束回合 {int(first_done.sum())}/{NUM_ENVS}  "
                  f"其中成功 {int(first_success.sum())}")
        if first_done.all():
            break

n_done = int(first_done.sum())
n_succ = int(first_success.sum())
n_fail = int(first_fail.sum())

print("\n=== 首个回合结局统计 ===")
print(f"已结束回合数     : {n_done}/{NUM_ENVS}")
print(f"成功抵达2026平台 : {n_succ} ({100.0 * n_succ / NUM_ENVS:.1f}%)")
print(f"失败(摔倒/超时)  : {n_fail} ({100.0 * n_fail / NUM_ENVS:.1f}%)")
if NUM_ENVS - n_done:
    print(f"到{MAX_STEPS}步仍未结束 : {NUM_ENVS - n_done}")

if n_succ:
    ss = steps_to_done[first_success]
    print(f"\n成功回合耗时(步) : 中位数={int(np.median(ss))}  "
          f"最快={int(ss.min())}  最慢={int(ss.max())}")
    print(f"成功回合耗时(秒) : 中位数={np.median(ss) * 0.01:.2f}s  最快={ss.min() * 0.01:.2f}s")

print(f"\n环境累计计数(含重开的后续回合): finished={env.episodes_finished} "
      f"succeeded={env.episodes_succeeded} failed_by_fall={env.episodes_failed_by_fall}")
