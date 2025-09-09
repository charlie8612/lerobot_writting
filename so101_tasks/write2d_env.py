# so101_tasks/write2d_env.py
# 只依賴 numpy、gymnasium
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _circle(n: int = 200, r: float = 0.30, cx: float = 0.50, cy: float = 0.50) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    return np.stack([r * np.cos(t) + cx, r * np.sin(t) + cy], axis=1)  # (N,2) in [0,1]^2


def _flatten_strokes(strokes: List[np.ndarray]) -> np.ndarray:
    if not strokes:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(strokes, axis=0).astype(np.float32)


def _as_strokes(target: Optional[np.ndarray], strokes: Optional[List[np.ndarray]]) -> List[np.ndarray]:
    """
    統一資料型別：
    - target: (N,2) → 當作單一筆劃
    - strokes: List[(Ni,2)]
    - 兩者皆 None → 預設圓形
    """
    if strokes is not None and len(strokes) > 0:
        out = [np.asarray(s, np.float32).reshape(-1, 2) for s in strokes if len(s) > 0]
        return out if out else [np.asarray(_circle(), np.float32)]
    if target is not None and len(target) > 0:
        return [np.asarray(target, np.float32).reshape(-1, 2)]
    return [np.asarray(_circle(), np.float32)]


class SO101Write2DEnv(gym.Env):
    """
    極簡 2D 寫字環境（可直接跑；觀測相容你的 wrapper 習慣）

    狀態（內部）：
      - pos ∈ [0,1]^2、pen ∈ {0,1}、(stroke_idx, wp_idx) 指向目前目標點
    動作（agent）：
      - action = [ax, ay, pen_cmd]；ax,ay ∈ [-1,1]，每步位移 = step_gain * [ax,ay]
      - pen_cmd ∈ [0,1]（>=0.5 視為落筆）。若 auto_lift_between_strokes=True，筆劃間會自動抬筆 lift_steps 步。
    觀測（Dict）：
      - "agent_pos": (3,) = [x, y, pen]  ← 你的 wrapper 用 obs["agent_pos"][:2]
      - "goal": (2,) 目前 waypoint
      - "wp_index": (1,) 目前 flatten 後的 waypoint 索引
      - "progress": (1,) 0~1
      - "dist_to_goal": (1,)
    回饋：
      - 基礎：-dist_to_goal
      - 若抵達半徑 tol 內並推進 waypoint → +progress_bonus
    終止條件：
      - terminated：到達最後 waypoint 且距離 < tol
      - truncated：步數達 max_episode_steps
    記錄：
      - self.trajectory 會存每一步的 t,x,y,pen，便於後續度量
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: Optional[np.ndarray] = None,
        strokes: Optional[List[np.ndarray]] = None,
        *,
        tol: float = 0.02,
        progress_bonus: float = 0.2,
        dt: float = 0.05,
        step_gain: float = 0.01,
        max_episode_steps: int = 400,
        auto_lift_between_strokes: bool = True,
        lift_steps: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # 目標筆劃
        self.strokes: List[np.ndarray] = _as_strokes(target, strokes)
        self.stroke_lens = [len(s) for s in self.strokes]
        self.fp = _flatten_strokes(self.strokes)  # flatten points (G,2)
        assert self.fp.shape[0] >= 1, "Empty target strokes."

        # 超參
        self.tol = float(tol)
        self.progress_bonus = float(progress_bonus)
        self.dt = float(dt)
        self.step_gain = float(step_gain)
        self.max_episode_steps = int(max_episode_steps)
        self.auto_lift = bool(auto_lift_between_strokes)
        self.lift_steps_cfg = int(max(0, lift_steps))

        # 觀測／動作空間（相容你的 wrapper）
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(low=np.array([0.0, 0.0, 0.0], np.float32),
                                        high=np.array([1.0, 1.0, 1.0], np.float32),
                                        shape=(3,), dtype=np.float32),
                "goal": spaces.Box(low=np.array([0.0, 0.0], np.float32),
                                   high=np.array([1.0, 1.0], np.float32),
                                   shape=(2,), dtype=np.float32),
                "wp_index": spaces.Box(low=0, high=self.fp.shape[0]-1, shape=(1,), dtype=np.int32),
                "progress": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "dist_to_goal": spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0], np.float32),
                                       high=np.array([1.0, 1.0, 1.0], np.float32),
                                       shape=(3,), dtype=np.float32)

        # 狀態
        self._t = 0.0
        self._step_count = 0
        self._pos = np.zeros(2, np.float32)
        self._pen = 0
        self._wp = 0
        self._stroke_idx = 0
        self._stroke_wp_bounds: List[Tuple[int, int]] = []
        acc = 0
        for L in self.stroke_lens:
            self._stroke_wp_bounds.append((acc, acc + L - 1))  # 各筆劃在 flatten 路徑上的起訖索引
            acc += L
        self._lift_left = 0
        self.trajectory: List[Dict[str, float]] = []

    # -------- gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pos = self.fp[0].copy().astype(np.float32)
        self._pen = 0
        self._wp = 0
        self._stroke_idx = 0
        self._t = 0.0
        self._step_count = 0
        self._lift_left = 0
        self.trajectory = []
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        a = np.asarray(action, np.float32).reshape(-1)
        if a.shape[0] != 3:
            raise ValueError("action shape must be (3,) = [ax, ay, pen_cmd]")
        ax, ay = float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
        pen_cmd = float(np.clip(a[2], 0.0, 1.0))

        # 自動斷筆（筆劃間抬筆）
        if self.auto_lift and self._lift_left > 0:
            self._pen = 0
            self._lift_left -= 1
        else:
            self._pen = 1 if pen_cmd >= 0.5 else 0

        # 位置更新（限制在紙面內）
        self._pos = np.clip(self._pos + self.step_gain * np.array([ax, ay], np.float32), 0.0, 1.0)

        # 目標距離與推進
        goal = self.fp[self._wp]
        dist = float(np.linalg.norm(self._pos - goal))
        reward = -dist

        progressed = False
        while dist < self.tol and self._wp < len(self.fp) - 1:
            self._wp += 1
            progressed = True
            goal = self.fp[self._wp]
            dist = float(np.linalg.norm(self._pos - goal))

        if progressed:
            reward += self.progress_bonus
            # 若剛好跨越筆劃邊界，安排自動抬筆
            if self.auto_lift:
                # 若 _wp 是某筆劃的起點，代表剛結束上一筆劃
                for si, (lo, hi) in enumerate(self._stroke_wp_bounds):
                    if self._wp == lo and si > 0:
                        self._lift_left = self.lift_steps_cfg
                        break

        # 成功：最後 waypoint 內距離 < tol
        done_success = (self._wp >= len(self.fp) - 1) and (dist < self.tol)

        # 記錄
        self.trajectory.append({"t": self._t, "x": float(self._pos[0]), "y": float(self._pos[1]), "pen": int(self._pen)})

        # 時間／步數
        self._t += self.dt
        self._step_count += 1

        terminated = bool(done_success)
        truncated = bool(self._step_count >= self.max_episode_steps)
        return self._get_obs(), float(reward), terminated, truncated, self._get_info(done_success=done_success, dist=dist)

    # -------- 便利存取 --------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        goal = self.fp[self._wp]
        progress = 0.0 if len(self.fp) <= 1 else self._wp / float(len(self.fp) - 1)
        obs = {
            "agent_pos": np.array([self._pos[0], self._pos[1], float(self._pen)], dtype=np.float32),
            "goal": goal.astype(np.float32),
            "wp_index": np.array([self._wp], dtype=np.int32),
            "progress": np.array([progress], dtype=np.float32),
            "dist_to_goal": np.array([float(np.linalg.norm(self._pos - goal))], dtype=np.float32),
        }
        return obs

    def _get_info(self, *, done_success: Optional[bool] = None, dist: Optional[float] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "success": bool(done_success) if done_success is not None else False,
            "time": float(self._t),
            "dt": float(self.dt),
            "wp_index": int(self._wp),
            "n_waypoints": int(len(self.fp)),
            "stroke_index": int(self._stroke_idx),  # 保留欄位，若要擴充可追蹤
        }
        if dist is not None:
            info["min_dist"] = float(dist)
        return info

    # 外部可取用的執行結果
    def get_trajectory(self) -> List[Dict[str, float]]:
        return list(self.trajectory)

    def get_target_strokes(self) -> List[np.ndarray]:
        return [s.copy() for s in self.strokes]

    # 允許動態替換筆劃
    def set_strokes(self, strokes: List[np.ndarray]):
        self.strokes = _as_strokes(None, strokes)
        self.stroke_lens = [len(s) for s in self.strokes]
        self.fp = _flatten_strokes(self.strokes)
        self._stroke_wp_bounds = []
        acc = 0
        for L in self.stroke_lens:
            self._stroke_wp_bounds.append((acc, acc + L - 1))
            acc += L
        # 重新 reset
        self.reset()


# 可做為 gym entry_point 的工廠函式
def make_env(**kwargs) -> SO101Write2DEnv:
    """
    gym.make('gym_hil/SO101Write2D-v0') 會呼叫這個工廠。
    你也可以傳入：
      - target=(N,2) 或 strokes=List[(Ni,2)]
      - tol, progress_bonus, dt, step_gain, max_episode_steps, auto_lift_between_strokes, lift_steps
    """
    return SO101Write2DEnv(**kwargs)
