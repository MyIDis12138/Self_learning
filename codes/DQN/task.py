
from math import gamma
import gym
import random
import numpy as np

from DQN import DQN

cfg = {
    "episodes": 6000,
    "n_time_step": 1000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay": 100000
}
agent_dict= {
        "batch_size": 128,
        "gamma": 0.99,
        "learning_rate": 0.002,
        "buffer_size": 10000,
        "update_frequency":20
    }


env = gym.make('CartPole-v1')
s = env.reset(seed=1)   

agent = DQN(
    env=env,
    batch_size=agent_dict["batch_size"],
    buffer_sizes=agent_dict["buffer_size"],
    gamma=agent_dict["gamma"],
    learning_rate=agent_dict["learning_rate"],
    update_frequency=agent_dict["update_frequency"]
)

reward_buffer = np.empty(shape=cfg["episodes"])

STPES_BEFORE = 5
for episode_i in range(cfg["episodes"]):
    episode_reward = 0
    for step_i in range(cfg["n_time_step"]):

        epsilon = np.interp(
            episode_i*cfg["episodes"] + step_i, 
            [0, cfg["epsilon_decay"]], 
            [cfg["epsilon_start"], cfg["epsilon_end"]]
            )
        
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.select_action(s) 

        s_, r, done, info,_ = env.step(a)
        a = np.array(a).copy()
        agent.memory.add(obs=s, action=a, done=done, reward=r, obs_=s_,infos=info) 
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            reward_buffer[episode_i] = episode_reward
            break

        #compute targets
        if step_i>STPES_BEFORE:
            agent.update() 
    print(f"Episode: {episode_i}, reward: {episode_reward}")






        


