import sys; sys.path.insert(0,"/home/taco/xarm")
import gymnasium as gym, so101_tasks
env = gym.make("gym_hil/SO101Write2D-LeRobot-v0")
obs, _ = env.reset()
print("state6:", obs["state6"].shape, "obs.state:", obs["observation.state"].shape, "agent_pos:", obs["agent_pos"].shape)
# 期望輸出：state6 (6,), obs.state (6,), agent_pos (2,)
