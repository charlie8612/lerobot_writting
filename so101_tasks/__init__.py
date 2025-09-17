# In /home/taco/xarm/so101_tasks/__init__.py
from gymnasium.envs.registration import register

# --- 我們只註冊唯一需要的 3D 環境 ---
register(
    id="gym_hil/PandaWriteKeyboard-v0",
    entry_point="so101_tasks.panda_write_keyboard:make_env",
    max_episode_steps=400,
)

# --- 將所有舊的、可能造成干擾的環境註冊全部註解掉 ---
# register(
#     id="gym_hil/SO101Write-v0",
#     entry_point="so101_tasks.so101_write_env:SO101WriteEnv",
#     max_episode_steps=400,
# )
#
# register(
#     id="gym_hil/SO101Write2D-Circle-v0",
#     entry_point="so101_tasks.write2d_env:make_env",
#     max_episode_steps=400,
# )
#
# register(
#     id="gym_hil/SO101Write2D-LeRobot-v0",
#     entry_point="so101_tasks.lerobot_write2d_wrapper:make_env",
#     max_episode_steps=400,
# )

print("--- [SO101 TASKS] Custom __init__.py loaded: Only PandaWriteKeyboard-v0 is registered. ---")