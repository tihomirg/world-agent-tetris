import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris

# create and reset the tetris environment
env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
observation, info = env.reset(seed=1337)

terminated = False

# loop until we loose
while not terminated:
    
    env.render()
    # get a random action from the action space
    action = env.action_space.sample()

    # apply action to the current state
    observation, reward, terminated, truncated, info = env.step(action)

    key = cv2.waitKey(100)

env.close()
print("Game Over!")