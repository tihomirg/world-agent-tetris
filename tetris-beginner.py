import cv2
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
    action = env.action_space.sample()

    # apply action to the current state
    observation, reward, terminated, truncated, info = env.step(action)

    key = cv2.waitKey(200)

print(f"Observation: \n{observation}")
print(f"Observation shape: \n{observation.shape}")
env.close()
print("Game Over!")