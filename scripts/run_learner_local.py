#!/usr/bin/env python3
import sys, traceback, types

# 讓 Python 先找你本機的 lerobot 與任務包
sys.path.insert(0, "/home/taco/so101/lerobot/src")
sys.path.insert(0, "/home/taco/xarm")

# 註冊環境（雖然 Learner 不一定會建 env，但保險）
import so101_tasks  # 觸發 __init__.py 的 register
print("[run_learner_local] so101_tasks imported", flush=True)

# ---- (A) SAC features 防呆：state(6) / action(3) ----
from lerobot.configs.types import PolicyFeature, FeatureType
import lerobot.policies.sac.configuration_sac as sac_conf

def _patched_validate_features(self):
    if not getattr(self, "input_features", None) or "observation.state" not in self.input_features:
        self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    if not getattr(self, "output_features", None) or "action" not in self.output_features:
        self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    return None

sac_conf.SACConfig.validate_features = _patched_validate_features
print("[run_learner_local] SACConfig.validate_features monkey-patched", flush=True)

# ---- (B) 關掉 policy instance 上的 normalizer，避免沒有 dataset stats 時出錯 ----
import lerobot.policies.factory as pol_factory

_orig_make_policy = pol_factory.make_policy

def _identity_forward(self, x, *args, **kwargs):
    return x

def _bind_identity_forward(module):
    module.forward = types.MethodType(_identity_forward, module)

def _disable_normalizers_on(policy):
    n = 0
    for name, module in policy.named_modules():
        if "normaliz" in module.__class__.__name__.lower():
            _bind_identity_forward(module)
            n += 1
    print(f"[run_learner_local] Disabled {n} normalizer module(s) on policy", flush=True)

def _make_policy_patched(*args, **kwargs):
    policy = _orig_make_policy(*args, **kwargs)
    _disable_normalizers_on(policy)
    return policy

pol_factory.make_policy = _make_policy_patched
print("[run_learner_local] make_policy() patched to disable normalizers on instance", flush=True)

# ---- (C) 呼叫官方 Learner CLI ----
from lerobot.scripts.rl.learner import train_cli

if __name__ == "__main__":
    print("[run_learner_local] about to call train_cli()", flush=True)
    try:
        train_cli()
        print("[run_learner_local] train_cli() returned", flush=True)
    except SystemExit as e:
        print(f"[run_learner_local] train_cli() SystemExit: {e.code}", flush=True)
        raise
    except Exception as e:
        print("[run_learner_local] train_cli() raised an exception:", e, flush=True)
        traceback.print_exc()
        raise
