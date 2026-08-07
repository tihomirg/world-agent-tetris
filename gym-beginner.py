import gymnasium as gym

# make environment and render every step for human to see it
env = gym.make("CartPole-v1", render_mode="human")

# Action space
print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation


# first observation
observation, info = env.reset()

print(f"Starting observation {observation}")

episode_over = False
total_reward = 0

# loop and play in the environment
while not episode_over:

    # chose a random action
    # action space contains all the possible actions of the agent in the environment
    action = env.action_space.sample()

    # apply the action to the environment and current state
    observation, reward, terminated, truncated, info = env.step(action)

    # after taking the action the reward can be positive if we are getting closer or negative if we are getting further from the goal
    total_reward += reward

    # terminated means that we reached the losing state (we lost)
    # truncated means we reached the set time limit 
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")

#at the end we need to clos the environment
env.close()