import gymnasium as gym
import numpy as np

class SO101WriteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        image_obs: bool = True,
        render_mode: str = "rgb_array",
        use_gripper: bool = False,
        gripper_penalty: float = 0.0,
        img_size=(128, 128),
        step_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.use_gripper = use_gripper
        self.gripper_penalty = float(gripper_penalty)

        self.img_size = tuple(img_size)
        self.step_scale = float(step_scale)

        # 動作: dx, dy, dz, pen_toggle
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        H, W = self.img_size
        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)
            }),
            "agent_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "agent_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "pen": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # 目標軌跡（示例）
        t = np.linspace(0, 2*np.pi, 200)
        self.target_traj = np.stack([0.3*np.cos(t)+0.5, 0.3*np.sin(t)+0.5], axis=1)

        self.reset(seed=None)

    def _obs(self):
        H, W = self.img_size
        rgb = np.zeros((H, W, 3), dtype=np.uint8)  # 先用空白圖（之後可疊畫軌跡）
        obs = {
            "images": {"front": rgb},
            "agent_pos": self.ee_state.astype(np.float32),      # 直接把 ee_state 當作 agent_pos
            "agent_vel": self.ee_vel.astype(np.float32),        # 上一步到這一步的差分
            "pen": np.array([1.0 if self.pen_down else 0.0], dtype=np.float32),
        }
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 末端狀態: x,y,z,roll,pitch,yaw
        self.ee_state = np.array([0.5, 0.2, 0.1, 0, 0, 0], dtype=np.float32)
        self.prev_ee_state = self.ee_state.copy()
        self.ee_vel = np.zeros_like(self.ee_state, dtype=np.float32)

        self.pen_down = False
        self.step_count = 0
        self._drawn_points = []
        obs = self._obs()
        info = {"target_traj": self.target_traj}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        dx, dy, dz, pen = action
        self.prev_ee_state = self.ee_state.copy()

        self.ee_state[:3] += self.step_scale * np.array([dx, dy, dz], dtype=np.float32)
        self.ee_state[:3] = np.clip(self.ee_state[:3], 0.0, 1.0)

        self.ee_vel = self.ee_state - self.prev_ee_state

        self.pen_down = pen > 0.0
        if self.pen_down:
            self._drawn_points.append(self.ee_state[:2].copy())

        p = self.ee_state[:2]
        dists = np.linalg.norm(self.target_traj - p[None, :], axis=1)
        min_dist = float(dists.min())
        reward = -min_dist
        if self.use_gripper and self.pen_down:
            reward += self.gripper_penalty

        terminated = False
        self.step_count += 1
        truncated = self.step_count >= 400
        info = {"min_dist": min_dist, "success": min_dist < 0.015}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self._obs()["images"]["front"]
