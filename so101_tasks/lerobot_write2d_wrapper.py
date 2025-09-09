# so101_tasks/lerobot_write2d_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .write2d_env import make_env as make_base_env  # 你原本的 2D 環境工廠

class ToLeRobotState(gym.ObservationWrapper):
    """
    將原本 obs 轉成 LeRobot 期望的單鍵：
      observation.state = [x, y, gx, gy, pen, progress]  (皆在 [0,1] 範圍內；pen ∈ {0,1})
    - 若 obs["agent_pos"] 只有 [x,y]，則嘗試從 obs["pen"] 取筆狀態；沒有就設成 0
    - 若欠缺 goal/progress，就補 0（但建議你的基礎環境本來就提供）
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=np.zeros(6, dtype=np.float32),
                high=np.ones(6, dtype=np.float32),
                shape=(6,), dtype=np.float32
            )
        })

    def observation(self, obs):
        # agent position: [x, y] 或 [x, y, pen]
        ap = np.asarray(obs.get("agent_pos"), dtype=np.float32).reshape(-1)
        if ap.shape[0] == 2:
            pen = np.asarray(obs.get("pen", 0.0), dtype=np.float32).reshape([])
            ap = np.array([ap[0], ap[1], float(pen)], dtype=np.float32)
        elif ap.shape[0] >= 3:
            ap = ap[:3].astype(np.float32)
        else:
            raise ValueError(f"agent_pos shape unexpected: {ap.shape}")

        # goal: [gx, gy]
        goal = np.asarray(obs.get("goal", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if goal.shape[0] < 2:
            goal = np.pad(goal, (0, 2 - goal.shape[0]), constant_values=0.0)
        goal = goal[:2].astype(np.float32)

        # progress: scalar
        prog = np.asarray(obs.get("progress", 0.0), dtype=np.float32).reshape(())
        prog = float(prog)

        state6 = np.array([ap[0], ap[1], goal[0], goal[1], ap[2], prog], dtype=np.float32)
        return {"observation.state": state6}

def make_env(**kwargs) -> gym.Env:
    """
    與 base 工廠同參數；同時把 LeRobot 會多塞的 kwargs 拿掉，避免 __init__ 不接受而報錯。
    """
    # 這幾個是 gym_manipulator 可能塞進來但你環境不吃的
    kwargs.pop("image_obs", None)
    kwargs.pop("render_mode", None)
    kwargs.pop("use_gripper", None)
    kwargs.pop("gripper_penalty", None)

    base = make_base_env(**kwargs)
    return ToLeRobotState(base)
