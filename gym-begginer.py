import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

print(f"Starting observation {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # chose an action
    action = env.action_space.sample()

    # take the action and see the new observation
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += 1
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()