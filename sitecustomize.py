import importlib
try:
    importlib.import_module("so101_tasks")
    print("[sitecustomize] so101_tasks imported (envs registered)")
except Exception as e:
    print("[sitecustomize] warning: failed to import so101_tasks:", e)
