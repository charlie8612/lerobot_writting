# so101_tasks/panda_write_keyboard.py
import gymnasium as gym
import numpy as np

# --------- 小工具：幾條簡單的目標軌跡 ---------
def _circle(n=200, r=0.30, cx=0.50, cy=0.50):
    t = np.linspace(0, 2*np.pi, n, dtype=np.float32)
    return np.stack([r*np.cos(t)+cx, r*np.sin(t)+cy], axis=1)  # (N,2) in [0,1]^2

def _line(n=200, x0=0.2, y0=0.5, x1=0.8, y1=0.5):
    t = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([x0 + t*(x1-x0), y0 + t*(y1-y0)], axis=1)

# --------- 只覆寫「回饋/成功」，其他觀測/視窗/操控完全沿用 gym_hil ---------
class WriteRewardWrapper(gym.Wrapper):
    """
    將 Panda*Keyboard 環境包裝成「臨摹筆畫」任務：
    - 目標軌跡 target_traj: (N,2) 紙面座標，已正規化到 [0,1]^2
    - pen_down：優先用夾爪狀態；沒有就用 Z 低於 z_contact 當作「落筆」
    - 回饋：越靠近目標越好；可選擇要求必須落筆才給分
    - 成功：coverage(ε) 達到門檻 或 連續多步 min_dist < ε
    """
    def __init__(self, env, target_traj, tol=0.02, progress_bonus=0.2):
            super().__init__(env)
            self.target = np.asarray(target_traj, np.float32).reshape(-1, 2)
            self.tol = float(tol)                    # 抵達 waypoint 的半徑
            self.progress_bonus = float(progress_bonus)
            self.i = 0                               # 目前目標 waypoint 索引
    
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.i = 0
        return obs, info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        xy = obs["agent_pos"][:2].astype(np.float32)

        # 與目前 waypoint 的距離
        d_vec = self.target[self.i] - xy
        dist = float(np.linalg.norm(d_vec))

        # 基礎：越近越好
        reward = -dist

        # 如果進入半徑 → 推進到下一個 waypoint 並給 progress 獎勵
        progressed = False
        while dist < self.tol and self.i < len(self.target) - 1:
            self.i += 1
            progressed = True
            d_vec = self.target[self.i] - xy
            dist = float(np.linalg.norm(d_vec))
        if progressed:
            reward += self.progress_bonus

        done_success = (self.i >= len(self.target) - 1 and dist < self.tol)

        info = dict(info or {})
        info.update({"min_dist": dist, "wp_index": int(self.i), "success": bool(done_success)})
        return obs, reward, term, trunc, info

# --------- 任務工廠：造官方 Panda 鍵盤環境，套上寫字回饋 ---------
def make_env(**kwargs):
    # 造官方鍵盤環境（帶視窗/相機/操控），kwargs 由 lerobot 傳入（image_obs/render_mode 等）
    base = gym.make("gym_hil/PandaPickCubeKeyboard-v0", **kwargs)
    # 目標：先用圓；之後你要字/筆畫，只要把這一行換成你的點列即可
    traj = _circle()
    return WriteRewardWrapper(base, target_traj=traj)