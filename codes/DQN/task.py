import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path


import string
import gym
import random
import numpy as np
from DQN import DQN

cfg = {
    #"env_name": "MountainCar-v0",
    "env_name": "MountainCar-v0",
    "episodes": 20,
    "n_time_step": 1000,
    "epsilon_start": 0.95,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "eval_epsisode": 20
}
agent_dict= {
        "batch_size": 64,
        "gamma": 0.95,
        "learning_rate": 0.0001,
        "buffer_size": 10000,
        "update_frequency":30,
        "hidden_dim": 256,
        "layer_num": 3
    }

# TODO logger tensorboard


def train(env: gym.Env, agent:DQN, cfg):
    s,_ = env.reset(seed=1)  
    reward_buffer = []

    for episode_i in range(cfg["episodes"]):
        episode_reward = 0
        for step_i in range(cfg["n_time_step"]):

            epsilon = np.interp(
                episode_i*cfg["episodes"] + step_i, 
                [0, cfg["epsilon_decay"]], 
                [cfg["epsilon_start"], cfg["epsilon_end"]]
                )
            
            random_sample = random.random()

            if random_sample > epsilon:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s) 
            
            s_, r, done, _, info = env.step(a)
            #print(a)
            a = np.array(a).copy()
            agent.memory.add(
                obs=s, 
                action=a, 
                done=done, 
                reward=r, 
                obs_=s_,
                infos=info
            ) 
            s = s_
            episode_reward += r
            
            agent.update()

            if done:
                s,_ = env.reset(seed=1)
                break
        
        if (len(reward_buffer)>1 and episode_reward >= max(reward_buffer)):
            name = cfg["env_name"]+"_best.pt"
            agent.save_model(curr_path, name)
        
        reward_buffer.append(episode_reward) 
        print(f"Episode: {episode_i}, reward: {episode_reward}")

    env.close()
    print("Finished!")


def eval(env_name: string, agent:DQN, eval_rounds:int):
    env = gym.make(env_name, render_mode='human')
    name = cfg["env_name"]+"_best.pt"
    agent.load_model(curr_path, name)
    s,_ = env.reset()
    total_reward = 0
    for i in range(eval_rounds):
        while True:
            a = agent.select_action(s)
            s_, r, done, info, _ = env.step(a) 
            env.render()
            total_reward += r
            if done:
                print(f"Evaluation round:{i} total reward: {total_reward}")
                s,_ = env.reset()
                total_reward = 0
                break
    env.close()


def main():
    env_name = cfg["env_name"]#, new_step_api=True) 
    
    env = gym.make(env_name)
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
    train(env,agent,cfg)
    env.close()
    eval(env_name, agent, cfg["eval_epsisode"])

if __name__=="__main__":
    main()
    




