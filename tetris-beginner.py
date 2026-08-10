import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiBinary 
from tetris_gymnasium.envs.tetris import Tetris
from gymnasium import ObservationWrapper

# Wrapper that will transform the original observation into a binary (20, 10) space 
class BinaryObservationSpaceWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = MultiBinary([20, 10])

    def observation(self, observation):
        return observation["board"][:20, 4:14] > 0

"""
Actions mapping:

move_left 0
move_right 1
move_down 2
rotate_clockwise 3
rotate_counterclockwise 4
hard_drop 5
swap 6
no-operation 7
"""
"""
Here we will focus only onto the following actions:
0: move_left 0
1: move_right 1
2: rotate_clockwise 3
3: hard_drop 5
4: no-operation 7
"""
actions = {
    0: 0,
    1: 1,
    2: 3,
    3: 5,
    4: 7}

actionNames = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "hard_drop",
    6: "swap",
    7: "no-operation"}

# create and reset the tetris environment
env_old = gym.make("tetris_gymnasium/Tetris", render_mode="human")
env = BinaryObservationSpaceWrapper(env_old)

observation, info = env.reset(seed=1337)
print(f"Observation space: {env.observation_space}")
print(f"Observation: \n{observation}")
print(f"Observation shape: \n{observation.shape}")


terminated = False

# loop until we loose
while not terminated:
    
    env.render()
    # get a random action from the action space
    # action = env.action_space.sample()
    action = actions[np.random.choice(len(actions))]

    print(f"Action: {actionNames[action]}")

    # apply action to the current state
    observation, reward, terminated, truncated, info = env.step(action)

    key = cv2.waitKey(150)

print(f"Observation: \n{observation}")
print(f"Observation shape: \n{observation.shape}")
env.close()
print("Game Over!")