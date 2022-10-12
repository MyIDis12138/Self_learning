import gym
env = gym.make("LunarLander-v2")
print(env.action_space)
env.close()