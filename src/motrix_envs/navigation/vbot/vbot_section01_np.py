# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection01EnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section01", "np")
class VBotSection01Env(NpEnv):
    """
    VBot在Section01地形上的导航任务
    继承自NpEnv，使用VBotSection01EnvCfg配置
    """
    _cfg: VBotSection01EnvCfg
    
    def __init__(self, cfg: VBotSection01EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        self._init_terrain_sensing()
        self._init_reward_helpers()

        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：68维 = 48(本体状态) + 12(足端: 接触/离地间隙/竖直速度 x4足) + 8(前方地形高度采样)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_range = 0.1  # 随机生成范围：±0.1m（0.2m×0.2m区域）
    
        # 导航统计计数器
        self.navigation_stats_step = 0

        # 回合结局累计统计（跨 reset 存活，供评估脚本读取真实成功率）
        self.episodes_finished = 0
        self.episodes_succeeded = 0
        self.episodes_failed_by_fall = 0
        # 逐环境的"上一回合结局"：0=尚未结束 1=成功抵达 2=失败。
        # info 里的字段会被 _reset_done_envs 清掉，评估脚本在 step() 返回后
        # 读不到结局，所以必须另存一份在环境对象上。
        self.last_episode_outcome = np.zeros(num_envs, dtype=np.int8)
        self.episodes_completed_per_env = np.zeros(num_envs, dtype=np.int64)
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._ground_geoms = self._collect_ground_geoms()
        self._init_termination_contact()
        self._init_foot_contact()

    def _collect_ground_geoms(self) -> list[int]:
        """
        收集所有"非机器人"的可碰撞 geom，作为地面/地形集合。

        section01 的地形来自 0126_C_section01.xml 等外部碰撞模型，其中绝大多数 geom
        在 XML 里没有 name（271 个 geom 中只有 29 个具名），因此原先按名字前缀
        (cfg.asset.ground_subtree = "C_") 匹配地面的做法一个都匹配不到，
        ground_geoms 恒为空，基座接触检测被整体禁用。

        这里改为反向筛选：排除机器人自身的 geom，再排除纯视觉 geom
        (collision_group == 0)，剩下的即为地形。
        """
        robot_geom_names = set(self._cfg.asset.foot_names)
        for name in self._model.geom_names:
            if name and name.startswith("collision_"):
                robot_geom_names.add(name)

        ground_geoms = []
        num_hfield = 0
        for idx in range(self._model.num_geoms):
            geom = self._model.get_geom(idx)
            if geom.collision_group == 0:
                continue  # 纯视觉 geom，不参与碰撞
            if geom.name in robot_geom_names:
                continue  # 机器人自身
            ground_geoms.append(idx)
            if geom.hfield is not None:
                num_hfield += 1

        print(f"[Info] 地面 geom: {len(ground_geoms)} 个（其中 hfield 高度场 {num_hfield} 个）")
        return ground_geoms

    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞对"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on

        pairs = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
                continue
            pairs.extend([base_geom_idx, ground_idx] for ground_idx in self._ground_geoms)

        if pairs:
            self.termination_contact = np.array(pairs, dtype=np.uint32)
            self.num_termination_check = len(pairs)
            print(
                f"[Info] 终止接触检测: {len(termination_contact_names)}个基座geom × "
                f"{len(self._ground_geoms)}个地面geom = {self.num_termination_check}个检测对"
            )
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")

    def _init_foot_contact(self):
        """初始化足端接触检测：每只脚 × 全部地面geom，查询后按脚聚合"""
        foot_names = self._cfg.asset.foot_names
        self.num_foot_check = len(foot_names)
        self.num_ground_geoms = len(self._ground_geoms)

        pairs = []
        for foot_name in foot_names:
            try:
                foot_idx = self._model.get_geom_index(foot_name)
            except Exception as e:
                print(f"[Warning] 无法找到足端geom '{foot_name}': {e}")
                self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
                return
            pairs.extend([foot_idx, ground_idx] for ground_idx in self._ground_geoms)

        self.foot_contact_check = np.array(pairs, dtype=np.uint32)
        print(
            f"[Info] 足端接触检测: {self.num_foot_check}只脚 × "
            f"{self.num_ground_geoms}个地面geom = {len(pairs)}个检测对"
        )

    def query_foot_contact(self, data: mtx.SceneData) -> np.ndarray:
        """返回 (num_envs, 4) 的足端触地布尔量，每只脚对所有地面geom取或"""
        num_envs = data.shape[0]
        if self.foot_contact_check.shape[0] == 0:
            return np.zeros((num_envs, self.num_foot_check), dtype=bool)
        cquery = self._model.get_contact_query(data)
        hit = cquery.is_colliding(self.foot_contact_check)
        return hit.reshape(num_envs, self.num_foot_check, self.num_ground_geoms).any(axis=2)

    def _get_foot_height_and_clearance(self, data: mtx.SceneData) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (foot_z, clearance)，均为 (num_envs, 4)。

        foot_z 是四只脚的世界高度，clearance = foot_z - 该点地面高度（用
        _sample_terrain_height 估计）。观测里的足端离地间隙、以及奖励里
        swing_foot_height / drop_leg_catchup 等摆腿相关项都用这份数据，
        统一在这里算一次，避免重复查询 geom pose。
        """
        num_envs = data.shape[0]
        foot_z = np.zeros((num_envs, 4), dtype=np.float32)
        clearance = np.zeros((num_envs, 4), dtype=np.float32)
        for i, geom_idx in enumerate(self._foot_geom_indices):
            pose = np.asarray(self._model.get_geom(geom_idx).get_pose(data))
            ground_h = self._sample_terrain_height(pose[:, :2])
            foot_z[:, i] = pose[:, 2]
            clearance[:, i] = pose[:, 2] - ground_h
        return foot_z, clearance

    def _init_terrain_sensing(self):
        """
        准备"前方地形高度采样"(8维)与"足端离地间隙"要用到的地形查询数据。

        section01 赛道只有起伏崎岖区 (-1.5 < Y < 1.5) 铺了真实的 hfield 高度场
        （geom C1_V_Adixing_Plane，绑定 hfield "C1_hfield_terrain"，实测起伏
        最高约 0.277m，与赛道设计文档一致）。其余路段（起步平台/落差坎/15°
        上坡/2026平台）是纯几何拼接，没有 hfield，用实测边界拼一个分段闭式
        高度公式：15°上坡从 Y=2.0 开始线性升高，到 Y=6.8296 接上平台，平台
        高度 1.2941m —— 这个值与场景里 C2_V_Bdixing_Plane 的世界 z 坐标
        (1.2941) 完全对得上，两处独立测量互相印证。
        """
        c1_geom_idx = self._model.get_geom_index("C1_V_Adixing_Plane")
        c1_hfield = self._model.get_geom(c1_geom_idx).hfield
        self._c1_height_matrix = np.asarray(c1_hfield.height_matrix, dtype=np.float32)
        self._c1_bound = np.asarray(c1_hfield.bound, dtype=np.float32)  # [xmin,ymin,zmin,xmax,ymax,zmax]
        self._c1_nrow = c1_hfield.nrow
        self._c1_ncol = c1_hfield.ncol

        self._slope_start_y = 2.0
        self._slope_end_y = 6.8296
        self._platform_height = 1.2941

        # 机身前方采样距离：0.2~1.6m，共8个点
        self._terrain_sample_dists = np.array(
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], dtype=np.float32
        )

        self._foot_geom_indices = [
            self._model.get_geom_index(name) for name in self._cfg.asset.foot_names
        ]

    def _sample_terrain_height(self, xy: np.ndarray) -> np.ndarray:
        """
        查询任意世界坐标 (x, y) 处的地面高度（米），全程向量化，支持任意前导维度。

        崎岖区内部用真实 hfield 双线性插值；崎岖区之外用分段闭式公式
        （起步/落差坎段视为 0，上坡段线性爬升，平台段常数）。
        """
        x = xy[..., 0]
        y = xy[..., 1]

        closed_form = np.zeros_like(y, dtype=np.float32)
        on_slope = (y >= self._slope_start_y) & (y < self._slope_end_y)
        on_platform = y >= self._slope_end_y
        slope_height = np.tan(np.deg2rad(15.0)) * (y - self._slope_start_y)
        closed_form = np.where(on_slope, slope_height, closed_form)
        closed_form = np.where(on_platform, self._platform_height, closed_form)

        b = self._c1_bound
        in_hfield = (x >= b[0]) & (x <= b[3]) & (y >= b[1]) & (y <= b[4])

        ncol, nrow = self._c1_ncol, self._c1_nrow
        col_f = np.clip((x - b[0]) / (b[3] - b[0]) * (ncol - 1), 0, ncol - 1)
        row_f = np.clip((y - b[1]) / (b[4] - b[1]) * (nrow - 1), 0, nrow - 1)
        col0 = np.floor(col_f).astype(np.int64)
        row0 = np.floor(row_f).astype(np.int64)
        col1 = np.minimum(col0 + 1, ncol - 1)
        row1 = np.minimum(row0 + 1, nrow - 1)
        fc = col_f - col0
        fr = row_f - row0

        m = self._c1_height_matrix
        h00, h01 = m[row0, col0], m[row0, col1]
        h10, h11 = m[row1, col0], m[row1, col1]
        bilinear = (
            h00 * (1 - fr) * (1 - fc)
            + h01 * (1 - fr) * fc
            + h10 * fr * (1 - fc)
            + h11 * fr * fc
        )

        return np.where(in_hfield, bilinear, closed_form).astype(np.float32)

    def _compute_foot_and_terrain_obs(
        self, data: mtx.SceneData, robot_xy: np.ndarray, robot_heading: np.ndarray, contacts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算观测里新增的 12(足端) + 8(前方地形) 维。

        足端 12 维 = 4只脚 x [接触标志, 离地间隙(m), 竖直速度(m/s)]。
        物理引擎的 contact query 只返回布尔碰撞，没有真实的三轴接触力
        传感器（vbot.xml 里没有为脚定义 touch/force 传感器，cfg.py 里那句
        "足部接触力传感器名称" 的注释对应不上任何实际存在的传感器），
        所以没有用假数据凑三轴力，改用这三个真实可测、且后续摆腿/落差
        奖励项也用得上的量。

        contacts 由调用方传入（而不是这里自己再查一次），因为奖励函数的
        feet_air_time 统计也需要同一份接触结果，避免每步重复做两次接触查询。
        """
        num_envs = robot_xy.shape[0]

        contact = contacts.astype(np.float32)  # (N,4)
        foot_z, clearance = self._get_foot_height_and_clearance(data)
        vert_vel = np.zeros((num_envs, 4), dtype=np.float32)
        for i, geom_idx in enumerate(self._foot_geom_indices):
            vel = np.asarray(self._model.get_geom(geom_idx).get_linear_velocity(data))
            vert_vel[:, i] = vel[:, 2]
        foot_obs = np.stack([contact, clearance, vert_vel], axis=-1).reshape(num_envs, 12)

        dirs = np.stack([np.cos(robot_heading), np.sin(robot_heading)], axis=-1)  # (N,2)
        sample_xy = (
            robot_xy[:, np.newaxis, :]
            + self._terrain_sample_dists[np.newaxis, :, np.newaxis] * dirs[:, np.newaxis, :]
        )  # (N,8,2)
        raw_height = self._sample_terrain_height(sample_xy)  # (N,8)
        terrain_obs = np.clip(raw_height / self._platform_height, 0.0, 1.0).astype(np.float32)

        return foot_obs.astype(np.float32), terrain_obs

    def _advance_waypoints(
        self, waypoint_idx: np.ndarray, robot_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据机器人当前位置推进路径点索引。

        进入当前目标点 waypoint_reach_threshold 范围内即视为到达并立刻切换到
        下一个点，不要求在点上停留；到达最后一个点后索引保持不变。
        返回 (推进后的waypoint_idx, 推进后对应的目标点xy, 是否已到达最终目标,
        本步是否刚到达了推进前的那个目标点——不管是不是最后一个点)。
        """
        cmd_cfg = self._cfg.commands
        waypoints = np.asarray(cmd_cfg.waypoint_targets, dtype=np.float32)  # (7, 2)
        last_idx = len(waypoints) - 1

        target_xy = waypoints[waypoint_idx]
        distance = np.linalg.norm(target_xy - robot_xy, axis=1)
        reached_current = distance < cmd_cfg.waypoint_reach_threshold
        can_advance = reached_current & (waypoint_idx < last_idx)
        waypoint_idx = np.where(can_advance, waypoint_idx + 1, waypoint_idx)

        # 索引推进后立即用新目标重新计算，当步就对下一个点生效
        target_xy = waypoints[waypoint_idx]
        distance = np.linalg.norm(target_xy - robot_xy, axis=1)
        final_reached = (waypoint_idx == last_idx) & (distance < cmd_cfg.waypoint_reach_threshold)

        return waypoint_idx, target_xy, final_reached, reached_current

    def _segment_speed_caps(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        按机器人当前 Y 坐标所在路段，返回 (线速度上限, 角速度上限)。

        线速度上限数值来自赛道设计文档："普通路段 0.90 / 崎岖区 0.45 / 落差段
        0.225 / 坡道 0.63"（m/s）。文档只给了这一档前进速度上限，没有单独拆
        前进/侧向分量——本控制器的 desired_vel_xy 本来就是直接指向目标点的
        世界系向量，不做前进/侧向分解，所以这里统一当作"线速度模长上限"处理，
        用一个整体缩放因子把向量按比例缩短，方向不变。

        角速度上限文档没给具体数值，按线速度收紧的比例插值给出（越难走的路段
        转向也收得越紧），数值本身可调，不是从文档抄来的。
        """
        rough_lo, rough_hi = float(self._c1_bound[1]), float(self._c1_bound[4])
        in_rough = np.logical_and(y >= rough_lo, y <= rough_hi)
        in_drop = np.logical_and(y > rough_hi, y < self._slope_start_y)
        in_slope = np.logical_and(y >= self._slope_start_y, y < self._slope_end_y)

        lin_cap = np.full_like(y, 0.90, dtype=np.float32)
        lin_cap = np.where(in_rough, 0.45, lin_cap)
        lin_cap = np.where(in_drop, 0.225, lin_cap)
        lin_cap = np.where(in_slope, 0.63, lin_cap)

        ang_cap = np.full_like(y, 1.0, dtype=np.float32)
        ang_cap = np.where(in_rough, 0.8, ang_cap)
        ang_cap = np.where(in_drop, 0.6, ang_cap)
        ang_cap = np.where(in_slope, 0.9, ang_cap)

        return lin_cap, ang_cap

    def _compute_waypoint_velocity_command(
        self,
        robot_xy: np.ndarray,
        robot_heading: np.ndarray,
        target_xy: np.ndarray,
        final_reached: np.ndarray,
        prev_filtered_command: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        用简单 P 控制器把"当前位置 -> 当前路径点"的位移转成期望线速度/角速度
        指令，按当前所在路段做分级限速，再做一阶低通滤波（时间常数
        waypoint_cmd_smooth_tau）使指令平滑，避免路径点切换瞬间指令跳变。

        限速在低通滤波之前对"原始目标指令"生效，而不是滤波之后再硬截断：
        这样跨路段边界时（比如从平台冲进崎岖区），目标值先一步降下来，低通
        滤波器负责把实际输出平滑地拉低，不会出现速度瞬间打断的突兀感。

        返回 (未滤波的期望线速度，供箭头可视化使用, 滤波后的3维指令[vx,vy,yaw_rate])。
        """
        cmd_cfg = self._cfg.commands
        position_error = target_xy - robot_xy
        desired_vel_xy = np.clip(position_error * cmd_cfg.waypoint_lin_kp, -1.0, 1.0)

        lin_cap, ang_cap = self._segment_speed_caps(robot_xy[:, 1])
        speed = np.linalg.norm(desired_vel_xy, axis=1)
        shrink = np.minimum(1.0, lin_cap / np.maximum(speed, 1e-6))
        desired_vel_xy = desired_vel_xy * shrink[:, np.newaxis]

        desired_vel_xy = np.where(final_reached[:, np.newaxis], 0.0, desired_vel_xy)

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(
            heading_to_movement > np.pi, heading_to_movement - 2 * np.pi, heading_to_movement
        )
        heading_to_movement = np.where(
            heading_to_movement < -np.pi, heading_to_movement + 2 * np.pi, heading_to_movement
        )
        desired_yaw_rate = np.clip(heading_to_movement * cmd_cfg.waypoint_ang_kp, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(final_reached, 0.0, desired_yaw_rate)

        raw_command = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        dt = self._cfg.ctrl_dt
        tau = cmd_cfg.waypoint_cmd_smooth_tau
        alpha = 1.0 - np.exp(-dt / tau)
        filtered_command = (alpha * raw_command + (1.0 - alpha) * prev_filtered_command).astype(np.float32)

        return desired_vel_xy, filtered_command

    def _init_reward_helpers(self):
        """
        准备奖励函数要用到的固定量：髋关节索引、关节软限位、以及"非法部位接触"
        （大腿/小腿等躯干附属结构碰到地面）的碰撞对。
        """
        actuator_names = list(self._model.actuator_names)
        self.hip_indices = [i for i, name in enumerate(actuator_names) if name and "hip" in name]

        # 12个真实关节的软限位：从模型硬限位（joint_limits 前3列是 target_x/y/yaw 的占位，
        # 之后12列才是FR/FL/RR/RL的hip/thigh/calf，顺序与 actuator_names 一致）向内收 10%，
        # 避免策略一直顶着硬限位跑。
        lower, upper = self._model.joint_limits
        joint_lower = np.asarray(lower[3:], dtype=np.float32)
        joint_upper = np.asarray(upper[3:], dtype=np.float32)
        limit_center = 0.5 * (joint_lower + joint_upper)
        half_range = 0.5 * (joint_upper - joint_lower)
        self.soft_joint_lower_limits = limit_center - 0.9 * half_range
        self.soft_joint_upper_limits = limit_center + 0.9 * half_range

        # "非法接触" = 除了脚以外，机身/大腿/小腿等其他 collision_* 部位碰到地面。
        # 这类接触说明摔倒或蹭地，应当被扣分；terminate_after_contacts_on 里的
        # 头部/中段 geom 已经在 _init_termination_contact 里单独处理（直接终止），
        # 这里不重复计入，避免同一次接触被扣两次分。
        foot_names = set(self._cfg.asset.foot_names)
        terminate_names = set(self._cfg.asset.terminate_after_contacts_on)
        undesired_names = [
            name
            for name in self._model.geom_names
            if name and name.startswith("collision_") and name not in terminate_names
        ]
        pairs = []
        for name in undesired_names:
            if name in foot_names:
                continue
            geom_idx = self._model.get_geom_index(name)
            pairs.extend([geom_idx, ground_idx] for ground_idx in self._ground_geoms)

        if pairs:
            self.undesired_contact = np.array(pairs, dtype=np.uint32)
            self.num_undesired_contact_check = len(pairs)
            self._num_undesired_geoms = len(undesired_names)
        else:
            self.undesired_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_undesired_contact_check = 0
            self._num_undesired_geoms = 0

    def query_undesired_contacts(self, data: mtx.SceneData) -> np.ndarray:
        """返回 (num_envs,) 布尔量：机身非脚部部位是否碰到了地面。"""
        num_envs = data.shape[0]
        if self.num_undesired_contact_check == 0:
            return np.zeros(num_envs, dtype=bool)
        cquery = self._model.get_contact_query(data)
        hit = cquery.is_colliding(self.undesired_contact)
        return hit.reshape(num_envs, self._num_undesired_geoms, len(self._ground_geoms)).any(axis=(1, 2))

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]
        
        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 足端接触与腾空计时：每步只查一次，观测和奖励共用同一份结果
        contacts = self.query_foot_contact(data)
        state.info["feet_air_time"] = np.where(
            contacts, 0.0, state.info["feet_air_time"] + cfg.ctrl_dt
        ).astype(np.float32)
        state.info["contacts"] = contacts

        # 导航目标：7个路径点依次推进（到达即切换，不停留）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)

        waypoint_idx, target_position, reached_all, reached_this_step = self._advance_waypoints(
            state.info["waypoint_idx"], robot_position
        )
        state.info["waypoint_idx"] = waypoint_idx

        # goal_done 锁存：一旦走完最后一个路径点就置位并保持。
        # 它同时决定"本回合算成功"和"回合到此结束"（见 _compute_terminated），
        # 任务定义就是"抵达2026平台"，抵达即完成，不要求抵达后继续站立。
        state.info["goal_done"] = state.info["goal_done"] | reached_all

        desired_vel_xy, velocity_commands = self._compute_waypoint_velocity_command(
            robot_position,
            robot_heading,
            target_position,
            reached_all,
            state.info["filtered_velocity_commands"],
        )
        state.info["filtered_velocity_commands"] = velocity_commands

        # 目标标记可视化仍用 (x, y, yaw) 三元组，yaw 位不再参与导航计算
        pose_commands = np.concatenate(
            [target_position, np.zeros((target_position.shape[0], 1), dtype=np.float32)],
            axis=-1,
        )
        state.info["pose_commands"] = pose_commands
        
        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        foot_obs, terrain_obs = self._compute_foot_and_terrain_obs(
            data, robot_position, robot_heading, contacts
        )

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3   本体状态共48维
                foot_obs,           # 12  足端: 接触/离地间隙/竖直速度 x4足
                terrain_obs,        # 8   机身前方0.2~1.6m地形高度采样
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 68)
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 先算终止条件，摔倒惩罚要用到
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        # 再算奖励
        reward = self._compute_reward(
            data, state.info, velocity_commands, target_position, reached_all, reached_this_step, terminated
        )

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        终止条件分两类，语义完全不同，必须分开记账：

        - 失败终止：基座触地(摔倒)，外加数值异常(NaN/速度溢出)兜底。
        - 成功终止：goal_done，即走完最后一个路径点抵达2026平台。

        两者都走 terminated（而不是 truncated），因为回合确实到此结束、
        之后没有任何未来收益可言，价值函数不该做自举。区别在于：失败要吃
        termination 惩罚，成功不吃——见 _compute_reward 里对 fail_terminated
        的处理。把"抵达即结束"写进终止条件，任务定义才和赛段要求一致
        （抵达平台即完成），否则策略会被迫在终点上空转到超时。
        """
        data = state.data
        num_envs = data.shape[0]

        # 基座触地 = 摔倒
        if self.num_termination_check > 0:
            cquery = self._model.get_contact_query(data)
            hit = cquery.is_colliding(self.termination_contact)
            base_contact = hit.reshape(num_envs, self.num_termination_check).any(axis=1)
        else:
            base_contact = np.zeros(num_envs, dtype=bool)

        # 数值兜底：速度溢出或出现 NaN 时直接终止，避免污染 rollout
        _, _, root_linvel = self._extract_root_state(data)
        invalid = ~np.isfinite(root_linvel).all(axis=1)
        over_speed = np.sum(np.square(root_linvel[:, :2]), axis=1) > 1e8

        fail_terminated = base_contact | invalid | over_speed
        goal_done = state.info["goal_done"]
        terminated = fail_terminated | goal_done

        state.info["fail_terminated"] = fail_terminated
        state.info["base_contact"] = base_contact
        self._accumulate_episode_outcomes(state.info, terminated)
        return state.replace(terminated=terminated)

    def _accumulate_episode_outcomes(self, info: dict, terminated: np.ndarray):
        """
        在环境对象上累计"回合结局"统计，供评估脚本读取。

        必须记在环境对象上而不是 info 里：info 里的字段会在回合结束后被
        _reset_done_envs 重置，评估脚本在 step() 返回后已经读不到结局了。

        超时(truncated)由基类在 update_state 之后才计算，这里按同样的条件
        (steps+1 >= max_episode_steps) 提前判定，保证超时回合也被计入分母。
        """
        done = terminated.astype(bool)
        max_steps = self._cfg.max_episode_steps
        if max_steps:
            done = done | ((info["steps"] + 1) >= max_steps)
        if not np.any(done):
            return
        success = done & info["goal_done"]
        self.episodes_finished += int(done.sum())
        self.episodes_succeeded += int(success.sum())
        self.episodes_failed_by_fall += int((done & info["fail_terminated"]).sum())

        if done.shape[0] == self.last_episode_outcome.shape[0]:
            self.last_episode_outcome[done & success] = 1
            self.last_episode_outcome[done & ~success] = 2
            self.episodes_completed_per_env[done] += 1

    def _compute_reward(
        self,
        data: mtx.SceneData,
        info: dict,
        velocity_commands: np.ndarray,
        target_position: np.ndarray,
        reached_all: np.ndarray,
        reached_this_step: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """
        section01 五段地形（起步平台/崎岖区/落差坎/15°上坡/2026平台）的完整奖励。

        分三类：导航跟踪（跟指令走）、稳定性与控制正则（走得稳、动作不抽风）、
        步态与分段塑形（只在特定 Y 区间生效，专门治那一段的失败模式）。
        权重集中在 cfg.reward_config.scales，这里只算各项"原始值"，
        缺项/命中不到的 key 会被下面的求和循环自动跳过。
        """
        cfg = self._cfg
        scales = cfg.reward_config.scales
        num_envs = data.shape[0]
        dt = max(cfg.ctrl_dt, 1e-6)

        root_pos, root_quat, root_vel = self._extract_root_state(data)
        robot_position = root_pos[:, :2]
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        robot_heading = self._get_heading_from_quat(root_quat)
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)

        contacts = info["contacts"]  # (N,4) bool，update_state 里本步已经刷新过
        feet_air_time = info["feet_air_time"]  # (N,4) 秒
        foot_z, clearance = self._get_foot_height_and_clearance(data)

        # ================= 一、导航跟踪 =================
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        tracking_sigma = 0.25
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_lin_vel = np.exp(-lin_vel_error / tracking_sigma)
        tracking_ang_vel = np.exp(-ang_vel_error / tracking_sigma)

        command_speed_xy = np.linalg.norm(velocity_commands[:, :2], axis=1)
        active_move = (command_speed_xy > 0.05).astype(np.float32)
        cmd_dir = velocity_commands[:, :2] / np.maximum(command_speed_xy[:, np.newaxis], 1e-6)
        forward_speed = np.sum(base_lin_vel[:, :2] * cmd_dir, axis=1)
        forward_progress = np.clip(forward_speed, 0.0, 1.5) * active_move

        target_dir = position_error / np.maximum(distance_to_target[:, np.newaxis], 1e-6)
        tracking_goal_vel = np.clip(np.sum(base_lin_vel[:, :2] * target_dir, axis=1), -1.0, 1.0) * active_move

        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_err = desired_heading - robot_heading
        heading_err = np.where(heading_err > np.pi, heading_err - 2 * np.pi, heading_err)
        heading_err = np.where(heading_err < -np.pi, heading_err + 2 * np.pi, heading_err)
        tracking_yaw = np.exp(-np.square(heading_err) / 0.25)

        prev_distance = info["prev_distance_to_target"]
        target_progress = np.clip(prev_distance - distance_to_target, -0.2, 0.2)
        info["prev_distance_to_target"] = distance_to_target.astype(np.float32)

        reach_goal = reached_this_step.astype(np.float32)
        reach_all_goal = reached_all.astype(np.float32)

        # ================= 二、稳定性与控制正则 =================
        if self.num_termination_check > 0:
            cquery = self._model.get_contact_query(data)
            base_contact = (
                cquery.is_colliding(self.termination_contact)
                .reshape(num_envs, self.num_termination_check)
                .any(axis=1)
            )
        else:
            base_contact = np.zeros(num_envs, dtype=bool)

        lin_vel_z = np.square(base_lin_vel[:, 2])
        ang_vel_xy = np.sum(np.square(gyro[:, :2]), axis=1)
        orientation = np.sum(np.square(projected_gravity[:, :2]), axis=1)

        torques = np.asarray(data.actuator_ctrls, dtype=np.float32)
        torques_sq = np.sum(np.square(torques), axis=1)
        dof_vel_sq = np.sum(np.square(dof_vel), axis=1)

        last_dof_vel = info["last_dof_vel"]
        dof_acc = (dof_vel - last_dof_vel) / dt
        dof_acc_sq = np.sum(np.square(dof_acc), axis=1)

        action_rate = np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1)

        dof_pos_limits = np.sum(
            np.clip(dof_pos - self.soft_joint_upper_limits, 0.0, None)
            + np.clip(self.soft_joint_lower_limits - dof_pos, 0.0, None),
            axis=1,
        )

        body_speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        speed_deficit = np.clip(command_speed_xy - body_speed_xy, 0.0, None)
        anti_stall = speed_deficit * active_move

        undesired_contacts = self.query_undesired_contacts(data).astype(np.float32)

        # ================= 三、步态与分段塑形 =================
        # 各分段的 Y 边界都来自赛道实测（见 _init_terrain_sensing 的说明），
        # 不是随手拍的经验值：崎岖区用 hfield 的真实边界，上坡/平台用
        # slope_start_y / slope_end_y。
        FEET_AIR_TIME_TARGET = 0.45
        SWING_TARGET_H = 0.12
        y = robot_position[:, 1]
        rough_lo, rough_hi = float(self._c1_bound[1]), float(self._c1_bound[4])
        in_rough = np.logical_and(y >= rough_lo, y <= rough_hi).astype(np.float32)
        drop_zone = np.logical_and(y > 1.3, y <= 1.8).astype(np.float32)
        leg_drive_zone = np.logical_and(y >= 1.8, y < self._slope_end_y).astype(np.float32)
        slope_zone = np.logical_and(y >= self._slope_start_y, y < self._slope_end_y).astype(np.float32)

        first_contact = np.logical_and(feet_air_time > 0.0, contacts)
        feet_air_time_reward = (
            np.sum((feet_air_time - FEET_AIR_TIME_TARGET) * first_contact, axis=1) * active_move
        )

        laziest_leg_air = np.min(feet_air_time, axis=1)
        per_leg_swing = np.clip(laziest_leg_air, 0.0, FEET_AIR_TIME_TARGET) * active_move
        # per_leg_swing 的反面：四条腿里最不爱迈步的那条腿，离目标腾空时间差多少就扣多少
        leg_stale_penalty = np.clip(FEET_AIR_TIME_TARGET - laziest_leg_air, 0.0, FEET_AIR_TIME_TARGET) * active_move

        rear_air = feet_air_time[:, 2:4]  # RR, RL
        laziest_rear_air = np.min(rear_air, axis=1)
        rear_leg_stale_penalty = (
            np.clip(FEET_AIR_TIME_TARGET - laziest_rear_air, 0.0, FEET_AIR_TIME_TARGET) * active_move
        )

        c = contacts.astype(np.float32)  # 顺序 FR, FL, RR, RL
        diag1_match = 1.0 - np.abs(c[:, 0] - c[:, 3])  # FR vs RL
        diag2_match = 1.0 - np.abs(c[:, 1] - c[:, 2])  # FL vs RR
        gait_symmetry = 0.5 * (diag1_match + diag2_match) * active_move

        rear_airborne_balance = np.square(feet_air_time[:, 2] - feet_air_time[:, 3])
        rear_clearance_balance = np.square(clearance[:, 2] - clearance[:, 3])

        swing_mask = 1.0 - c  # 摆动腿=1
        swing_height_err = np.sum(swing_mask * np.abs(clearance - SWING_TARGET_H), axis=1)
        swing_foot_height = np.exp(-swing_height_err) * in_rough * active_move

        drop_leg_catchup = np.clip(laziest_leg_air, 0.0, FEET_AIR_TIME_TARGET) * drop_zone
        drop_pitch = np.square(projected_gravity[:, 0]) * drop_zone

        slope_leg_drive = np.clip(laziest_leg_air, 0.0, FEET_AIR_TIME_TARGET) * leg_drive_zone
        front_leg_air = np.minimum(feet_air_time[:, 0], feet_air_time[:, 1])  # FR, FL
        slope_front_drive = np.clip(front_leg_air, 0.0, FEET_AIR_TIME_TARGET) * leg_drive_zone

        slope_hip = np.zeros(num_envs, dtype=np.float32)
        if self.hip_indices:
            hip_idx = np.asarray(self.hip_indices, dtype=np.int64)
            hip_dev = dof_pos[:, hip_idx] - self.default_angles[hip_idx]
            hip_excess = np.clip(np.abs(hip_dev) - 0.2, 0.0, None)
            slope_hip = np.sum(np.square(hip_excess), axis=1) * slope_zone

        reward_terms = {
            "reach_all_goal": reach_all_goal,
            "reach_goal": reach_goal,
            "tracking_goal_vel": tracking_goal_vel,
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "tracking_yaw": tracking_yaw,
            "forward_progress": forward_progress,
            "target_progress": target_progress,
            # 只有失败终止(摔倒/数值异常)才吃这个罚分；成功抵达终点也会让
            # terminated=True，但那是完成任务，不能按摔倒罚。
            "termination": info["fail_terminated"].astype(np.float32),
            "base_contact": base_contact.astype(np.float32),
            "lin_vel_z": lin_vel_z,
            "ang_vel_xy": ang_vel_xy,
            "action_rate": action_rate,
            "torques": torques_sq,
            "dof_vel": dof_vel_sq,
            "dof_acc": dof_acc_sq,
            "anti_stall": anti_stall,
            "dof_pos_limits": dof_pos_limits,
            "undesired_contacts": undesired_contacts,
            "orientation": orientation,
            "feet_air_time": feet_air_time_reward,
            "per_leg_swing": per_leg_swing,
            "leg_stale_penalty": leg_stale_penalty,
            "rear_leg_stale_penalty": rear_leg_stale_penalty,
            "gait_symmetry": gait_symmetry,
            "rear_airborne_balance": rear_airborne_balance,
            "rear_clearance_balance": rear_clearance_balance,
            "swing_foot_height": swing_foot_height,
            "drop_leg_catchup": drop_leg_catchup,
            "drop_pitch": drop_pitch,
            "slope_leg_drive": slope_leg_drive,
            "slope_front_drive": slope_front_drive,
            "slope_hip": slope_hip,
        }

        reward = np.zeros(num_envs, dtype=np.float32)
        reward_contrib = {}
        for key, scale in scales.items():
            term = reward_terms.get(key)
            if term is None:
                continue
            term = np.nan_to_num(term.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
            contrib = float(scale) * term
            reward += contrib
            reward_contrib[key] = contrib

        info["Reward"] = reward_contrib
        return np.clip(reward, -100.0, 1000.0).astype(np.float32)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection01EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # 在高台中央小范围内随机生成位置
        # X, Y: 在spawn_center周围 ±spawn_range 范围内随机
        random_xy = np.random.uniform(
            low=-self.spawn_range,
            high=self.spawn_range,
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_xy  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)  # 使用配置的高度
        
        
        # 组合XYZ坐标
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz  # [x, y, z] 随机生成的位置
        
        # 路径点导航：所有新出生的环境都从第0个路径点开始
        waypoint_idx = np.zeros(num_envs, dtype=np.int64)
        target_positions = np.asarray(cfg.commands.waypoint_targets, dtype=np.float32)[waypoint_idx]
        pose_commands = np.concatenate(
            [target_positions, np.zeros((num_envs, 1), dtype=np.float32)], axis=1
        )
        filtered_velocity_commands = np.zeros((num_envs, 3), dtype=np.float32)
        
        # 归一化base的四元数（DOF 6-9）
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        
        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 导航目标：与update_state一致的路径点推进逻辑（新出生环境从第0点开始，
        # 这里再调用一次是为了处理"出生点恰好落在到达阈值内"的边界情况）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)

        waypoint_idx, target_position, reached_all, reached_this_step = self._advance_waypoints(
            waypoint_idx, robot_position
        )
        distance_to_target = np.linalg.norm(target_position - robot_position, axis=1)

        desired_vel_xy, filtered_velocity_commands = self._compute_waypoint_velocity_command(
            robot_position,
            robot_heading,
            target_position,
            reached_all,
            filtered_velocity_commands,
        )
        velocity_commands = filtered_velocity_commands

        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        contacts = self.query_foot_contact(data)

        foot_obs, terrain_obs = self._compute_foot_and_terrain_obs(
            data, robot_position, robot_heading, contacts
        )

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3   本体状态共48维
                foot_obs,           # 12  足端: 接触/离地间隙/竖直速度 x4足
                terrain_obs,        # 8   机身前方0.2~1.6m地形高度采样
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 68)
        
        info = {
            "pose_commands": pose_commands,
            "waypoint_idx": waypoint_idx,
            "filtered_velocity_commands": filtered_velocity_commands,
            "goal_done": np.zeros(num_envs, dtype=bool),
            "fail_terminated": np.zeros(num_envs, dtype=bool),
            "base_contact": np.zeros(num_envs, dtype=bool),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 统一使用min_distance机制
            # 新增：与locomotion一致的字段
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 上一步关节速度
            "contacts": contacts,  # 足部接触状态
            "feet_air_time": np.zeros((num_envs, self.num_foot_check), dtype=np.float32),  # 每只脚腾空累计时长
            "prev_distance_to_target": distance_to_target.copy(),  # 上一步到当前目标点的距离，用于 target_progress
        }
        
        return obs, info
    