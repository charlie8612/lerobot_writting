#!/usr/bin/env python3
import sys, traceback, types, numpy as np, torch

# 讓 Python 先找你本機的 lerobot 與任務包
sys.path.insert(0, "/home/taco/so101/lerobot/src")
sys.path.insert(0, "/home/taco/xarm")

# 註冊環境
import so101_tasks  # 觸發 __init__.py 的 register

# ---- (A) SAC features 防呆：state(6) / action(3) ----
from lerobot.configs.types import PolicyFeature, FeatureType
import lerobot.policies.sac.configuration_sac as sac_conf

def _patched_validate_features(self):
    if not getattr(self, "input_features", None) or "observation.state" not in self.input_features:
        self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(19,))}
    if not getattr(self, "output_features", None) or "action" not in self.output_features:
        self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    return None

sac_conf.SACConfig.validate_features = _patched_validate_features
print("[run_actor_local] SACConfig.validate_features monkey-patched", flush=True)

# ---- (B) 關閉 policy instance 上的 normalizer，避免沒有 dataset stats 時出錯 ----
import lerobot.policies.factory as pol_factory

_orig_make_policy = pol_factory.make_policy

def _identity_forward(self, x, *args, **kwargs):
    return x

def _bind_identity_forward(module):
    module.forward = types.MethodType(_identity_forward, module)

def _disable_normalizers_on(policy):
    n = 0
    for _, module in policy.named_modules():
        if "normaliz" in module.__class__.__name__.lower():
            _bind_identity_forward(module)
            n += 1
    print(f"[run_actor_local] Disabled {n} normalizer module(s) on policy", flush=True)

def _make_policy_patched(*args, **kwargs):
    policy = _orig_make_policy(*args, **kwargs)
    _disable_normalizers_on(policy)
    return policy

pol_factory.make_policy = _make_policy_patched
print("[run_actor_local] make_policy() patched to disable normalizers on instance", flush=True)

# ---- (C) 兼容 obs：如果環境直接輸出 observation.state，就略過預設的 preprocess_observation ----
import lerobot.envs.utils as env_utils
_orig_preprocess_observation = env_utils.preprocess_observation

def _preprocess_observation_flexible(observations):
    # 我們的 2D wrapper 直接給 {"observation.state": np.ndarray(6,)}
    if isinstance(observations, dict) and "observation.state" in observations:
        st = observations["observation.state"]
        if isinstance(st, np.ndarray):
            st_t = torch.from_numpy(st).float()
        else:
            st_t = torch.as_tensor(st, dtype=torch.float32)
        # 這裡就只回傳 policy 需要的鍵
        return {"observation.state": st_t}
    # 其他情況走原本流程（例如未來你切回含 agent_pos 的 obs）
    return _orig_preprocess_observation(observations)

env_utils.preprocess_observation = _preprocess_observation_flexible
print("[run_actor_local] preprocess_observation patched to accept observation.state", flush=True)

# ---- (D) 呼叫官方 actor CLI ----
from lerobot.scripts.rl.actor import actor_cli

if __name__ == "__main__":
    print("[run_actor_local] about to call actor_cli()", flush=True)
    try:
        actor_cli()
        print("[run_actor_local] actor_cli() returned", flush=True)
    except SystemExit as e:
        print(f"[run_actor_local] actor_cli() SystemExit: {e.code}", flush=True)
        raise
    except Exception as e:
        print("[run_actor_local] actor_cli() raised an exception:", e, flush=True)
        traceback.print_exc()
        raise
