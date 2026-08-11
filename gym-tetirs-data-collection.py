import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiBinary 
from tetris_gymnasium.envs.tetris import Tetris
from gymnasium import ObservationWrapper

# Hyperparameters
total_dataset_size = 5 # 10000 # number of (obs, action) -> (next_obs) pairs
dataset_file_name = "gym-tetris-dataset.npy"

# -------------------------------------------------------------------------------------------------------------------------------------------

# Wrapper that transforms the original observation into a binary (20, 10) space 
class BinaryObservationSpaceWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = MultiBinary([20, 10])

    def observation(self, observation):
        return observation["board"][:20, 4:14] > 0

# -------------------------------------------------------------------------------------------------------------------------------------------
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
We take a subset of actions:
0: move_left 0
1: move_right 1
2: rotate_clockwise 3
3: no-operation 7
"""
actions = {
    0: 0,
    1: 1,
    2: 3,
    3: 7}

actionNames = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "hard_drop",
    6: "swap",
    7: "no-operation"}

def chooseAction(actions):
    return actions[np.random.choice(len(actions))]

# -------------------------------------------------------------------------------------------------------------------------------------------

# create and reset the tetris environment
env_old = gym.make("tetris_gymnasium/Tetris", render_mode="human")
env = BinaryObservationSpaceWrapper(env_old)

terminated = True
data_size = 0

data = []

while total_dataset_size > data_size:

    # if we exit before 
    if terminated or truncated:
        obs, info = env.reset()
    
    env.render()

    # get a random action
    action = chooseAction(actions)

    #print(f"Action: {actionNames[action]}")

    # apply action to the current state
    next_obs, reward, terminated, truncated, info = env.step(action)

    key = cv2.waitKey(150)

    data.append({
        "obs": obs,
        "action": action,
        "next_obs": next_obs
    })

    obs = next_obs

    data_size += 1


env.close()

np.save(dataset_file_name, np.array(data))