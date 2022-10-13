import gym
import random
import numpy as np
from torch import save,load

from DQN import DQN

cfg = {
    "episodes": 500,
    "n_time_step": 1000,
    "epsilon_start": 0.5,
    "epsilon_end": 0.02,
    "epsilon_decay": 10000
}
agent_dict= {
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.002,
        "buffer_size": 10000,
        "update_frequency":30,
        "hidden_dim": 20,
        "layer_num": 5
    }




def train(env:gym.Env, agent:DQN, cfg):
    s = env.reset(seed=1)  
    reward_buffer = np.empty(shape=cfg["episodes"])

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
            #print(a)
            a = np.array(a).copy()
            agent.memory.add(obs=s, action=a, done=done, reward=r, obs_=s_,infos=info) 
            s = s_
            episode_reward += r
            
            agent.update()

            if done:
                s = env.reset()
                reward_buffer[episode_i] = episode_reward
                break
            
            if episode_reward>reward_buffer.max():
                save(agent.q_net,'best_pt')

        print(f"Episode: {episode_i}, reward: {episode_reward}")

    env.close()
    print("Finished!")


def eval(env:gym.Env, agent:DQN):
    agent.q_net = load('best_pt')
    s = env.reset()
    while True:
        a = agent.select_action(s)
        s_, r, done, info, _ = env.step(a) 
        env.render()

        if done:
            env.reset()
            break


def main():
    env = gym.make('CartPole-v1', new_step_api=True) 

    agent = DQN(
        env=env,
        batch_size=agent_dict["batch_size"],
        buffer_sizes=agent_dict["buffer_size"],
        gamma=agent_dict["gamma"],
        learning_rate=agent_dict["learning_rate"],
        update_frequency=agent_dict["update_frequency"],
        q_net_h_dim=agent_dict["hidden_dim"],
        q_net_layer_num=agent_dict["layer_num"]
    )

    print(agent.q_net)
    #train(env, agent, cfg)
    eval(env, agent)

if __name__=="__main__":
    main()
    




