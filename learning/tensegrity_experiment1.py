import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class TensegrityEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        distance_reward_weight=5.0,
        # forward_reward_weight=5.0,
        total_distance_reward_weight=3.0,
        velocity_reward_weight = 1.0,
        distance_increase_reward_weight=5.0,
        ctrl_cost_weight=0.01,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.1, 1.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            distance_reward_weight,
            # forward_reward_weight,
            total_distance_reward_weight,
            velocity_reward_weight,
            distance_increase_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._distance_reward_weight=distance_reward_weight
        # self._forward_reward_weight = forward_reward_weight
        self._total_distance_reward_weight = total_distance_reward_weight
        self._velocity_reward_weight  = velocity_reward_weight
        self._distance_increase_reward_weight=distance_increase_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.previous_distance_traveled = 0.0 #초기 이동거리 설정

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
            low=-np.inf, high=np.inf, shape=(650,), dtype=np.float64
        )
        

        MujocoEnv.__init__(
            self,
            "tensegrity_6_Bar.xml",  # XML 파일의 경로를 지정합니다.
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    # def step(self, action):
    #     xy_position_before = mass_center(self.model, self.data)
    #     self.do_simulation(action, self.frame_skip)
    #     xy_position_after = mass_center(self.model, self.data)

    #     xy_velocity = (xy_position_after - xy_position_before) / self.dt
    #     # x_velocity, y_velocity = xy_velocity

    #     # ctrl_cost = self.control_cost(action)

        
    #     # 이동거리 보상
    #     distance_traveled = np.linalg.norm(xy_position_after - xy_position_before)
    #     distance_reward = self._distance_reward_weight*distance_traveled

    #     # 총 이동 거리 보상 (시작점으로부터의 거리)
    #     total_distance = np.linalg.norm(xy_position_after - self.initial_xy_position)
    #     total_distance_reward = self._total_distance_reward_weight * total_distance
        
    #     # 속도 보상
    #     velocity_reward = self._velocity_reward_weight * np.linalg.norm(xy_velocity)

    #     healthy_reward = self.healthy_reward

    #     # 최종 보상 계산
    #     reward = distance_reward + total_distance_reward + velocity_reward + healthy_reward

    #       # 간단한 종료 조건 (너무 낮거나 높아지면 종료)
    #     terminated = bool(self.data.qpos[2] < 0.1 or self.data.qpos[2] > 1.0)

    #     observation = self._get_obs()
    #     info = {
    #     "reward_distance": distance_reward,
    #     "reward_total_distance": total_distance_reward,
    #     "reward_velocity": velocity_reward,
    #     # "reward_rotation": rotation_reward,
    #     "x_position": xy_position_after[0],
    #     "y_position": xy_position_after[1],
    #     "height": self.data.qpos[2],
    #     "distance_traveled": distance_traveled,   
    #     "total_distance": total_distance,
    #     }

    #     if self.render_mode == "human":
    #         self.render()
    #     return observation, reward, terminated, False, info
    
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        # 현재 스텝에서의 x축 이동 거리 계산
        x_distance_traveled = abs(xy_position_after[0] - xy_position_before[0])
        distance_reward = self._distance_reward_weight * x_distance_traveled

        # 초기 위치에서의 x축 총 이동 거리 계산
        total_x_distance = abs(xy_position_after[0] - self.initial_xy_position[0])
        total_distance_reward = self._total_distance_reward_weight * total_x_distance

        # 초기 위치에서부터의 x축 거리 증가 보상
        distance_increase = total_x_distance - self.previous_distance_traveled
        distance_increase_reward = self._distance_increase_reward_weight * distance_increase
        self.previous_distance_traveled = total_x_distance

        # x축 속도 보상 계산
        velocity_reward = self._velocity_reward_weight * abs(xy_velocity[0])

        # 자세 유지 보상 추가
        healthy_reward = self.healthy_reward

        # 최종 보상 계산
        reward = distance_reward + total_distance_reward + velocity_reward + distance_increase_reward + healthy_reward

        terminated = bool(self.data.qpos[2] < 0.1 or self.data.qpos[2] > 1.0)

        observation = self._get_obs()
        info = {
            "reward_distance": distance_reward,
            "reward_total_distance": total_distance_reward,
            "reward_velocity": velocity_reward,
            "reward_healthy": healthy_reward,
            "reward_distance_increase": distance_increase_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "height": self.data.qpos[2],
            "distance_traveled": x_distance_traveled,
            "total_distance": total_x_distance,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        # 초기 위치 설정
        self.initial_xy_position = mass_center(self.model, self.data)

        observation = self._get_obs()
        return observation
