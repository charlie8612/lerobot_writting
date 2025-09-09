from gymnasium.envs.registration import register

# 在 import so101_tasks 時自動註冊任務
register(
    id="gym_hil/SO101Write-v0",
    entry_point="so101_tasks.so101_write_env:SO101WriteEnv",
    max_episode_steps=400,   # 例如 40s * 10Hz control
)

register(
    id="gym_hil/PandaWriteKeyboard-v0",
    entry_point="so101_tasks.panda_write_keyboard:make_env",
    max_episode_steps=400,
)

register(
    id="gym_hil/SO101Write2D-Circle-v0",
    entry_point="so101_tasks.write2d_env:make_env",
    max_episode_steps=400,
)

register(
    id="gym_hil/SO101Write2D-LeRobot-v0",
    entry_point="so101_tasks.lerobot_write2d_wrapper:make_env",
    max_episode_steps=400,
)