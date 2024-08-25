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
            exclude_current_positions_from_observation=True,
            reset_noise_scale=1e-2,
            **kwargs,
    ):
        utils.Ezpickle.__init__(
            self,
            exclude_current_positions_from_observation,
            reset_noise_scale,
            **kwargs,
        )

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=np.inf, high=np.inf, shape=(650,),dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "tensegrity_12_Bar.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

     

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
    
    def step(self, action):
        #변수
        xypos_bf = mass_center(self.model, self.data) #prev
        self.do_simulation(action, self.frame_skip)
        xypos_af = mass_center(self.model, self.data) #curr

        #re_forward - x
        x_diff = xypos_af[0] - xypos_bf[0]
        re_forward = x_diff / self.dt

        #re_heading
        r_diff = xypos_bf - xypos_af
            # 이동 벡터 정규화 (길이를 1로 만듦)
        if np.linalg.norm(r_diff) != 0:
            heading_vec = r_diff / np.linalg.norm(r_diff)
        else:
            heading_vec = np.array([0, 0])  # 움직임이 없는 경우

        re_heading = 0.01 * np.dot(heading_vec, np.array([1,0]))
        if re_heading < 0.0 :
            re_heading = re_heading * 100

        #re_lane
        lane_diviation = xypos_af[1]
        re_lane = -abs(lane_diviation) * 0.5

        #comute re  
        reward = re_forward + re_heading + re_lane

        terminated  = bool(xypos_af - xypos_bf == 0 )
        observation = self._get_obs()
        info = {
            "re_forward": re_forward,
            "re_heading": re_heading,
            "re_lane": re_lane,
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
