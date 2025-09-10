# so101_tasks/panda_write_keyboard.py
# (請用以下內容完整替換)

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --------- 小工具：幾條簡單的目標軌跡 ---------
def _circle(n=200, r=0.15, cx=0.5, cy=0.4):
    """
    產生一個圓形軌跡。注意：Panda 環境的工作空間較大，
    所以半徑 r 需要調小一些，圓心也需要微調。
    """
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    # Panda 的 XY 平面是桌子，Z 軸是上下
    # 我們讓它在一個固定的高度 (z=0.1) 上畫圓
    z = np.full(n, 0.1, dtype=np.float32)
    return np.stack([r * np.cos(t) + cx, r * np.sin(t) + cy, z], axis=1)

# --------- 包裝器 1：修改獎勵機制，變成寫字任務 ---------
class WriteRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_traj, tol=0.02, progress_bonus=0.2):
        super().__init__(env)
        # 目標軌跡現在是 3D 的 (N, 3)
        self.target = np.asarray(target_traj, np.float32).reshape(-1, 3)
        self.tol = float(tol)
        self.progress_bonus = float(progress_bonus)
        self.i = 0  # 目前目標 waypoint 索引

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.i = 0
        return obs, info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        # agent_pos 現在是 3D 座標
        xyz = obs["agent_pos"][:3].astype(np.float32)

        # 與目前 waypoint 的距離
        dist = float(np.linalg.norm(self.target[self.i] - xyz))
        reward = -dist

        # 如果進入半徑 -> 推進到下一個 waypoint 並給獎勵
        progressed = False
        while dist < self.tol and self.i < len(self.target) - 1:
            self.i += 1
            progressed = True
            dist = float(np.linalg.norm(self.target[self.i] - xyz))
        
        if progressed:
            reward += self.progress_bonus

        done_success = (self.i >= len(self.target) - 1 and dist < self.tol)

        info = dict(info or {})
        info.update({"min_dist": dist, "wp_index": int(self.i), "success": bool(done_success)})
        # 我們回傳修改後的 reward，並沿用原始環境的 terminated 和 truncated
        return obs, reward, term, trunc, info

# --------- 包裝器 2：轉換觀測值格式以相容 LeRobot ---------
class PandaToLeRobotState(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # LeRobot 的 SAC 模型通常只看 agent 的狀態
        # 這裡我們使用 agent_pos (x,y,z,r,p,y) 作為輸入特徵
        state_shape = env.observation_space["agent_pos"].shape
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
            )
        })

    def observation(self, obs):
        # 從字典格式的觀測值中，只挑出 agent_pos 作為 state
        return {"observation.state": obs["agent_pos"].astype(np.float32)}

# --------- 任務工廠：串聯起所有元件 ---------
def make_env(**kwargs):
    print("\n\n>>> 正在載入 3D Panda 環境... <<<\n\n")
    # 1. 建立基礎的 3D Panda 環境
    base = gym.make("gym_hil/PandaPickCubeKeyboard-v0", **kwargs)
    
    # 2. 用 Wrapper 修改獎勵機制，讓它變成「跟隨圓圈軌跡」的任務
    reward_wrapped_env = WriteRewardWrapper(base, target_traj=_circle())
    
    # 3. 再用另一個 Wrapper 修改觀測值格式，讓它相容 LeRobot
    final_env = PandaToLeRobotState(reward_wrapped_env)
    
    return final_env