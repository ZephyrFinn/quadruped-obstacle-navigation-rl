# VBot 四足机器人越障导航 · Section01

四足机器人通过强化学习，自主走完一条 10 米长的障碍赛道：
**起步平台 → 起伏崎岖区 → 落差坎 → 15° 上坡 → 2026 平台**。

| 指标 | 本项目 | 同赛段公开方案 |
| --- | --- | --- |
| 成功率 | **94.1%** (241/256) | 87.5% |
| 完赛耗时中位数 | **10.08 s** | 13.9 s |
| 训练量 | 15,000 timesteps | 45,000 timesteps |

> 评估口径：256 并行环境、确定性动作、仅统计每个环境的第一个回合。
> 原始输出见 [`artifacts/eval_result.txt`](artifacts/eval_result.txt)。
> 与公开方案的对比为参考性对比 —— 赛道相同，但完赛判定细节可能有出入。

---

## 起点：官方 starter_kit 是一个空骨架

这一点决定了整个项目的工作量。官方给的 `vbot_section01_np.py`（679 行）里，
两个最核心的函数都是空的 —— 原文件完整保留在
[`reference/vbot_section01_np.ORIGINAL.py`](reference/vbot_section01_np.ORIGINAL.py)：

```python
def _compute_reward(self, data, info, velocity_commands):
    cfg = self._cfg
    # 计算总奖励
    reward = np.array([0])        # ← 奖励恒为 0，PPO 拿不到任何学习信号
    return reward

def _compute_terminated(self, state):
    data = state.data
    terminated = np.zeros(self._num_envs, dtype=bool)   # ← 摔倒永远不触发终止
    return state.replace(terminated=terminated)
```

第二个函数尤其致命：摔倒不结束回合，机器人躺在地上一直躺到超时。
而超时（truncated）在 PPO 里会把价值函数的估计接回去，
**等于告诉策略"躺平没有损失"**。

---

## 工作部分

设计规格来自公开参考方案，**缺陷定位、量化验证、口径设计是本项目的工作**：

| 内容 | 来源 |
| --- | --- |
| 68 维观测的维度构成、7 个路径点坐标、分段限速数值 | 公开参考方案的规格 |
| 33 项奖励的公式与权重 | 公开参考方案的实现 |
| PPO 超参、4096 并行环境 | starter_kit 自带 |
| **摔倒检测失效的定位与修复** | 本项目（两份参考文档均未记录此缺陷） |
| **足端观测的可测量替代方案** | 本项目 |
| **终止罚分语义修正** | 本项目（参考实现成功抵达时也照罚） |
| **确定性评估脚本与统计口径** | 本项目 |

### 独立定位的三处缺陷

**① 摔倒检测因 geom 名称匹配零命中而被完全禁用**

配置里按名称前缀 `ground_subtree = "C_"` 查找地面 geom。把加载后的
271 个 geom 全部 dump 出来才发现：只有 29 个有名字，**242 个是匿名的**
（地形来自外部碰撞模型 XML，里面的 geom 根本没起名字），前缀匹配到 **0 个**。

修法是反向排除：排掉机器人自身的 geom（`collision_` 开头 + 四只脚），
再排掉纯视觉 geom（`collision_group == 0`），剩下 164 个即为地形。
→ [`_collect_ground_geoms`](src/motrix_envs/navigation/vbot/vbot_section01_np.py)

**② 规格要求的传感器在模型里不存在**

规格写足端观测是"12 维接触力（4 足 × 3 轴）"。但 `vbot.xml` 里只有关节角度、
角速度、力矩三种传感器，**没有给脚装 touch/force 传感器**，物理引擎 API 也只有
`is_colliding`（返回布尔值），拿不到力。

没有编造三轴力凑维度，改用三个真实可测的量：**接触标志、离地间隙、竖直速度**。
这三个在后续的抬腿高度、落差跟腿奖励里正好用得上。

**③ 「抵达终点未终止回合」的任务定义缺陷**

评估发现 91% 的环境能走到终点，但其中 92% 后来又摔倒了。
第一反应是加一项"站稳奖励"，重训后**更差**（到达后摔倒率涨到 100%）。

对照公开实现逐行比对，才发现关键的一行：他们把"抵达终点"直接算作回合终止条件，
而本项目的终止条件只检查摔倒 —— 机器人到终点后回合还在跑，被迫在平台上
干站三千多步直到超时。任务要求是"抵达 2026 平台"，不是"抵达并站立 40 秒"，
**这个额外难度是代码自己强加的**。

修正后：成功率 **57.8% → 94.1%**。

> 顺带比参考实现更严谨的一处：参考实现的终止罚分对所有终止一视同仁，
> 成功抵达也吃 −10。本项目改为**只罚失败终止**。

---

## 为什么不看训练曲线

TensorBoard 里的 `Total reward` 是**回合累加值**，回合越长数字越大，
两个策略之间没法直接比较。项目中期就栽过一次：

无限速版本的 `slope_leg_drive`（上坡驱动奖励）累计 **980**，
加了限速的版本只有 **16**，看起来前者爬坡强得多。实际相反 ——
这项奖励要求"正在移动"，加了限速的版本走完全程后速度归零不再得分，
而无限速版本**从没走完过**，一直在坡上反复尝试反复触发，攒了海量"尝试分"。

**奖励高恰恰说明卡在坡上。**

所以判断依据全部来自 [`src/scripts/eval_policy.py`](src/scripts/eval_policy.py)：
确定性动作、256 环境、**仅统计每个环境的第一个回合**
（成功的回合结束更快、更快重开，统计累计次数会让成功率虚高）。

---

## 复现

本仓库只包含改动的源码与成果，不含赛道资产（约 265 MB，且授权归主办方）。

```bash
# 1. 官方框架（指定分支）
git clone --branch MotrixArena-S1 https://github.com/Motphys/MotrixLab.git
cd MotrixLab && uv sync --all-packages --extra skrl-torch

# 2. 赛道资产：下载 starter_kit 并按其说明解压到 motrix_envs/src/motrix_envs/
#    https://dist.bj.bcebos.com/motphys-arena/starter_kit.zip

# 3. 用本仓库的实现覆盖对应文件
cp src/motrix_envs/navigation/vbot/{vbot_section01_np.py,cfg.py} \
   MotrixLab/motrix_envs/src/motrix_envs/navigation/vbot/

# 4. 可视化确认地形加载正常
uv run scripts/view.py --env vbot_navigation_section01

# 5. 训练（4096 并行环境，约 75 分钟）
uv run scripts/train.py --env vbot_navigation_section01 --train-backend torch --seed 42

# 6. 评估（复现 94.1%）
uv run python eval_policy.py artifacts/best_agent.pt 256 4000

# 7. 看画面
uv run scripts/play.py --env vbot_navigation_section01 --num-envs 1 \
    --policy artifacts/best_agent.pt --seed 42
```

训练好的权重：[`artifacts/best_agent.pt`](artifacts/best_agent.pt)（3.1 MB）

---

## 技术要点速查

**观测 68 维**

| 组 | 维度 | 内容 |
| --- | --- | --- |
| 本体状态 | 48 | 线速度 3 · 角速度 3 · 重力投影 3 · 关节角 12 · 关节速度 12 · 上步动作 12 · 速度指令 3 |
| 足端 | 12 | 4 足 × [接触标志, 离地间隙, 竖直速度] |
| 前方地形 | 8 | 机身前方 0.2 ~ 1.6 m 的地面高度 |

前方地形采样从仿真的高度场矩阵做**双线性插值**取真实值（上坡段与平台段用闭式几何），
使策略在走到落差点之前就能"看见"地面变化，从被动反应转为主动预判。

**7 路径点课程分解**

```
(0,-0.60) → (0,1.20) → (0,2.25) → (0,4.00) → (0,6.00) → (0,7.00) → (0,7.80)
```

直接给十米外的终点是稀疏奖励问题 —— 训练初期随机动作永远够不着，学不动。
拆成 7 个关卡门（进入 0.45 m 即过关并切下一个），把长任务变成一串短任务。

**分段速度上限**

| 路段 | 起步/终点平台 | 崎岖区 | 落差坎 | 15° 上坡 |
| --- | --- | --- | --- | --- |
| 线速度上限 | 0.90 m/s | 0.45 m/s | 0.225 m/s | 0.63 m/s |

限速作用在低通滤波**之前**：跨路段边界时目标值先降，滤波器负责把实际输出
平滑拉低，避免速度硬拉断。这一步让"走完全程"从**从未发生**变为稳定发生。

---

## 训练迭代记录

| 轮次 | 改动 | 结果 |
| --- | --- | --- |
| v1 | 33 项奖励，无分段限速 | 15,000 步内**没有一个环境走完全程** |
| v2 | + 分段限速 | 首次出现完整走完全程的个体 |
| v3 | + 站稳奖励 | **失败** —— 到达后摔倒率 92.3% → 100% |
| v4 | 抵达即终止，删除 v3 改动 | **94.1%**，中位耗时 10.08 s |

v3 是一次方向性错误：把"任务定义写错"误判成了"平衡能力不足"。
保留在记录里，因为推翻它的过程比结果本身更说明问题。

---

## 来源与致谢

- 框架：[MotrixLab](https://github.com/Motphys/MotrixLab)（Apache-2.0），`MotrixArena-S1` 分支
- 赛道资产与 starter_kit：Motphys MotrixArena S1 赛事材料（本仓库不含，请从官方渠道获取）
- 观测规格、路径点坐标与奖励设计参考：[whatif218/MotrixArena-Section01](https://github.com/whatif218/MotrixArena-Section01)

本仓库中 `src/` 下的实现基于 MotrixLab 修改，遵循 Apache-2.0；
`reference/` 中的原始文件来自官方 starter_kit，仅作对照用途保留。
