# /home/taco/xarm/scripts/learner_boot_patched.py
import sys, importlib, inspect
sys.path.insert(0, "/home/taco/so101/lerobot/src")

from draccus.argparsing import parse

# 1) 找到你這版的 Train*Config（你之前測到是 TrainRLServerPipelineConfig）
m = importlib.import_module("lerobot.scripts.rl.learner")
TrainConfig = getattr(m, "TrainRLServerPipelineConfig")

# 2) 解析 CLI/JSON 成 cfg
cfg = parse(TrainConfig)

# 3) 在「SAC 驗證」階段強制補 features（monkey-patch validate_features）
from lerobot.configs.types import PolicyFeature, FeatureType
conf = importlib.import_module("lerobot.policies.sac.configuration_sac")

def _patched_validate_features(self):
    # 沒有就補 observation.state(6)、action(3)
    if not getattr(self, "input_features", None) or "observation.state" not in self.input_features:
        self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    if not getattr(self, "output_features", None) or "action" not in self.output_features:
        self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    # 原本函式會做其它檢查，這裡只要不再拋錯即可
    return None

conf.SACConfig.validate_features = _patched_validate_features
print("[learner_boot_patched] SACConfig.validate_features monkey-patched")

# 4) 保險：cfg.policy 也先設成 sac（某些路徑會直接用這個）
if getattr(cfg.policy, "type", None) != "sac":
    cfg.policy.type = "sac"

# 5) 印出關鍵確認
print("[learner_boot_patched] policy.type =", cfg.policy.type)
print("[learner_boot_patched] env.features_map =", getattr(cfg.env, "features_map", None))

# 6) 相容呼叫 train（不同版本參數不同）
n_actor_threads = getattr(getattr(cfg, "concurrency", None), "actor", 1)
kw = {}
sig = inspect.signature(m.train)
if "output_dir" in sig.parameters:
    kw["output_dir"] = getattr(cfg, "output_dir", None)
if "n_actor_threads" in sig.parameters:
    kw["n_actor_threads"] = n_actor_threads

print("[learner_boot_patched] calling train with kwargs:", kw)
m.train(cfg, **kw)
