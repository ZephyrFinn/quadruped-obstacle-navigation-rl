# 四足机器人越障导航

VBot 四足机器人在 MotrixArena S1 Section01 赛道上的导航策略。赛道全长约 10 m，
依次经过起步平台、起伏崎岖区、落差坎、15° 上坡，终点是 2026 平台。

![demo](docs/demo.gif)

256 环境确定性评估，成功率 94.1%（241/256），完赛耗时中位数 10.08 s。
原始输出见 [`artifacts/eval_result.txt`](artifacts/eval_result.txt)。

## 运行

赛道资产约 265 MB 且版权属主办方，没放进仓库，需要自己下载。

```bash
git clone --branch MotrixArena-S1 https://github.com/Motphys/MotrixLab.git
cd MotrixLab
uv sync --all-packages --extra skrl-torch

# 下载 starter_kit，按其说明把 navigation 资产解压到 motrix_envs/src/motrix_envs/
# https://dist.bj.bcebos.com/motphys-arena/starter_kit.zip

# 用本仓库的实现替换掉对应文件
cp <this-repo>/src/motrix_envs/navigation/vbot/*.py \
   motrix_envs/src/motrix_envs/navigation/vbot/
```

训练、评估、回放：

```bash
uv run scripts/train.py --env vbot_navigation_section01 --train-backend torch --seed 42
uv run python eval_policy.py artifacts/best_agent.pt 256 4000
uv run scripts/play.py --env vbot_navigation_section01 --num-envs 1 --policy artifacts/best_agent.pt
```

4096 并行环境，15000 timesteps，单卡 RTX 5880 Ada 约 75 分钟。
训练好的权重在 [`artifacts/best_agent.pt`](artifacts/best_agent.pt)。

## 起始代码的三个问题

官方 starter_kit 的 `vbot_section01_np.py` 是骨架，两个核心函数都没实现。
原文件保留在 [`reference/`](reference/vbot_section01_np.ORIGINAL.py) 下：

```python
def _compute_reward(self, data, info, velocity_commands):
    reward = np.array([0])
    return reward

def _compute_terminated(self, state):
    terminated = np.zeros(self._num_envs, dtype=bool)
    return state.replace(terminated=terminated)
```

除此之外还有两处没那么明显的。

**地面 geom 一个也匹配不上。** cfg 里用 `ground_subtree = "C_"` 按名字前缀找地面。
把加载后的 geom 全打出来才发现，271 个里有 242 个是匿名的——地形从外部碰撞模型
XML 加载，那些 geom 在 XML 里就没起名字。前缀匹配命中 0 个，摔倒检测从来没生效过。

改成反过来筛：去掉 `collision_` 开头的机器人 geom 和四只脚，再去掉
`collision_group == 0` 的纯视觉 geom，剩下 164 个就是地形。

**足端力传感器不存在。** 规格里足端观测是 12 维接触力（4 足 × 3 轴），
但 `vbot.xml` 只有关节角度、角速度、力矩三种传感器，脚上没装 touch/force，
引擎的 `is_colliding` 也只返回布尔值。

改用三个能实测的量：接触标志、离地间隙、竖直速度。同样 12 维，
而且后面做抬腿高度和落差跟腿奖励时正好都用得上。

## 实现要点

观测 68 维：

| | 维度 | 内容 |
| --- | --- | --- |
| 本体状态 | 48 | 线速度 3、角速度 3、重力投影 3、关节角 12、关节速度 12、上步动作 12、速度指令 3 |
| 足端 | 12 | 4 足 × [接触标志, 离地间隙, 竖直速度] |
| 前方地形 | 8 | 机身前方 0.2 ~ 1.6 m 的地面高度 |

前方地形这 8 维从仿真的高度场矩阵做双线性插值取真实值，上坡段和平台段用闭式几何算。
有了它，策略在还没走到落差点之前就知道前面地面要掉下去。

路径点：

```
(0,-0.60) → (0,1.20) → (0,2.25) → (0,4.00) → (0,6.00) → (0,7.00) → (0,7.80)
```

直接给十米外的终点是稀疏奖励问题，训练初期随机动作永远够不着。拆成 7 个关卡门，
进入 0.45 m 即过关并切下一个，速度指令由 P 控制器算出来再过一阶低通（时间常数 0.25 s）。

分段限速：

| 路段 | 平台 | 崎岖区 | 落差坎 | 上坡 |
| --- | --- | --- | --- | --- |
| 线速度上限 | 0.90 | 0.45 | 0.225 | 0.63 m/s |

限速加在低通滤波之前，跨路段时目标值先降，由滤波器把实际输出平滑拉低。
加上这个之后才第一次出现走完全程的个体。

奖励 33 项，分导航跟踪、姿态与控制正则、步态与分段塑形三类。分段塑形只在特定路段生效，
比如 `drop_leg_catchup` 只在 Y 1.3~1.8 有效，专门治落差坎上"前腿出去后腿没跟上"。

## 评估

不看 TensorBoard 的 total reward。那是回合累加值，回合越长数字越大，两个策略之间没法直接比。
中间栽过一次：无限速版本的 `slope_leg_drive` 累计 980，限速版本只有 16，看着像前者爬坡强，
其实反了——这项奖励要求正在移动，限速版本走完全程后速度归零就不再得分，
而无限速版本从没走完过，一直卡在坡上反复触发，攒了一堆"尝试分"。

[`eval_policy.py`](src/scripts/eval_policy.py) 的口径：256 环境、确定性动作、
只统计每个环境的第一个回合。只统计第一个回合是因为成功的回合结束更快也重开更快，
按累计次数算会把成功率算高。

## 迭代记录

| | 改动 | 结果 |
| --- | --- | --- |
| v1 | 33 项奖励，无分段限速 | 15000 步内没有一个环境走完全程 |
| v2 | 加分段限速 | 首次出现走完全程的个体 |
| v3 | 加"到终点后站稳"奖励 | 更差，到达后摔倒率从 92.3% 涨到 100% |
| v4 | 抵达即终止，撤掉 v3 | 94.1%，中位耗时 10.08 s |

v3 走了弯路。当时评估发现 91% 能到终点、其中 92% 到了之后又摔，判断是平衡能力不够，
就加了站稳奖励，重训之后反而全摔。

对着公开实现逐行比对才发现是终止条件漏了一项：抵达终点应该直接结束回合，
而我这边只判断摔倒，所以机器人到终点后还得在平台上站三千多步等超时。
任务要求是抵达平台，不是抵达并站满 40 秒，这个难度是自己代码加上去的。

改完之后 57.8% → 94.1%。57.8% 是用新口径回测旧策略的结果；旧口径下那版报的是 91%，
统计的是"曾经进过终点范围"，包含摔倒重开之后才进的。换口径重测才有可比性。

顺带改了一处：参考实现的终止罚分对所有终止一视同仁，成功抵达也扣 −10，
这里改成只罚失败终止。

## 来源

- 框架 [MotrixLab](https://github.com/Motphys/MotrixLab)（Apache-2.0），`MotrixArena-S1` 分支
- 赛道资产与 starter_kit 来自 Motphys MotrixArena S1，本仓库不含
- 观测维度构成、路径点坐标、奖励权重参考了
  [whatif218/MotrixArena-Section01](https://github.com/whatif218/MotrixArena-Section01)

`src/` 下的实现基于 MotrixLab 修改，沿用 Apache-2.0。
`reference/` 里是官方 starter_kit 原文件，只作对照。
