# In so101_tasks/panda_write_keyboard.py
# (請用以下內容完整替換)

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --------- 全局常量：定義物理邊界 ---------
Z_CONTACT_THRESHOLD = 0.02  # Z 軸接觸高度 (單位：公尺)

# --------- 小工具：半圓軌跡 ---------
def _semicircle_in_front(n=100, r=0.2, cx=0.5, cy=0.35):
    t = np.linspace(0, np.pi, n, dtype=np.float32)
    x = r * np.cos(t) + cx
    y = r * np.sin(t) + cy
    z = np.full(n, Z_CONTACT_THRESHOLD, dtype=np.float32)
    return np.stack([x, y, z], axis=1)

# --------- 包裝器 1：將 3 維動作翻譯成 4 維 ---------
class ActionTranslatorWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.append(action, 0.0).astype(np.float32)

# --------- 包裝器 2：轉換觀測值格式，並追加「接觸」狀態 (最終修正版) ---------
class StateAndContactWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 原始的 agent_pos 是 6 維或 18 維，取決於框架的包裝
        # 我們直接取用它，並追加一個維度
        original_state_shape = env.observation_space["agent_pos"].shape
        new_state_dim = original_state_shape[0] + 1

        # 建立新的、19 維的觀測空間
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(new_state_dim,), dtype=np.float32
            )
        })

    def observation(self, obs):
        # 1. 將 `agent_pos` 作為基礎狀態 (這是之前缺失的關鍵步驟)
        #    lerobot 的 gym_manipulator 會將 6 維 agent_pos 擴充成 18 維
        base_state = obs["agent_pos"].astype(np.float32)
        
        # 2. 根據 Z 軸座標，判斷是否接觸
        current_z = base_state[2] # Z 軸是第 3 個元素
        is_in_contact = 1.0 if current_z <= Z_CONTACT_THRESHOLD else 0.0
        
        # 3. 將「接觸」狀態作為最後一個維度，追加到基礎狀態後面
        final_state = np.append(base_state, is_in_contact).astype(np.float32)
        
        # 4. 以 `lerobot` 期望的格式回傳
        return {"observation.state": final_state}

# --------- 包裝器 3：修改獎勵機制，使用「接觸」狀態 ---------
class ContactRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_traj, tol=0.02, progress_bonus=0.2):
        super().__init__(env)
        self.target = np.asarray(target_traj, np.float32).reshape(-1, 3)
        self.tol = float(tol)
        self.progress_bonus = float(progress_bonus)
        self.i = 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.i = 0
        return obs, info

    def step(self, action):
        # obs, _, term, trunc, info = self.env.step(action)
        # 我們不再需要傳遞底層的 info，以避免不一致
        obs, _, term, trunc, _ = self.env.step(action)

        full_state = obs["observation.state"]
        xyz = full_state[:3]
        is_in_contact = full_state[-1] > 0.5 # 接觸狀態是最後一個維度

        dist = float(np.linalg.norm(self.target[self.i] - xyz))
        reward = -dist

        if is_in_contact:
            while dist < self.tol and self.i < len(self.target) - 1:
                self.i += 1
                dist = float(np.linalg.norm(self.target[self.i] - xyz))
                reward += self.progress_bonus # 每推進一步都給獎勵

        done_success = (self.i >= len(self.target) - 1 and dist < self.tol and is_in_contact)

        # 建立一個全新的、結構固定的 info 字典
        # 這樣可以確保回傳給 lerobot 的資訊維度永遠一致
        info = {
            "min_dist": dist,
            "wp_index": int(self.i),
            "success": float(done_success),
            "in_contact": float(is_in_contact)
        }
        return obs, reward, term, trunc, info

# --------- 任務工廠：串聯起所有新的元件 (最終修正版) ---------
def make_env(**kwargs):
    # 1. 建立 lerobot 原始的 Panda 環境
    base_env = gym.make("gym_hil/PandaPickCubeKeyboard-v0", **kwargs)
    
    # 2. 套上我們的 3->4 維動作翻譯官
    action_wrapped_env = ActionTranslatorWrapper(base_env)
    
    # 3. 再套上我們新的、能處理 `agent_pos` 並追加接觸狀態的觀測值修改器
    obs_wrapped_env = StateAndContactWrapper(action_wrapped_env)

    # 4. 在最外層，套上我們新的、基於物理接觸的獎勵機制
    final_env = ContactRewardWrapper(obs_wrapped_env, target_traj=_semicircle_in_front())
    
    return final_env