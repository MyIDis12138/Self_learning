import sys,os
from tokenize import Triple
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path


import string
import gym
import random
import numpy as np
from DDQN import DQN


cfg = {
    #"env_name": "MountainCar-v0",
    "env_name": "CartPole-v1",
    "episodes": 500,
    "n_time_step": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_end": 450,
    "eval_epsisode": 20,
    "layer_num": 5
}
agent_dict= {
        "batch_size": 32,
        "gamma": 0.99,
        "learning_rate": 0.0001,
        "buffer_size": 100000,
        "update_frequency":30,
        "hidden_layers": [32,32],
    }

# TODO logger tensorboard


def train(env: gym.Env, agent:DQN, cfg):
    s,_ = env.reset()  
    reward_buffer = []
    done = False

    for episode_i in range(cfg["episodes"]):
        episode_reward = 0
        step_i=0
        s,_ = env.reset()
        #for step_i in range(cfg["n_time_step"]):
        while True:
            epsilon = np.interp(
                episode_i, 
                [0, cfg["epsilon_decay_end"]], 
                [cfg["epsilon_start"], cfg["epsilon_end"]]
                )

            if random.random() > epsilon:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s) 
            
            s_, r, done, _, info = env.step(a)
            #print(a)
            a = np.array(a)
            agent.memory.add(obs=s, action=a, done=done, reward=r, obs_=s_) 
            
            s = s_
            episode_reward += r
            agent.update()
            step_i+=1

            if done or step_i>cfg["n_time_step"]:
                break
        
        if (len(reward_buffer)>1 and episode_reward >= max(reward_buffer)):
            name = cfg["env_name"]+"_"+str(cfg["layer_num"])+"_best.pt"
            agent.save_model(curr_path, name)
        
        reward_buffer.append(episode_reward) 
        print(f"Episode: {episode_i}, reward: {episode_reward}")

    env.close()
    print("Finished!")


def eval(env_name: string, agent:DQN, cfg):
    env = gym.make(env_name)
    name = cfg["env_name"]+"_"+str(cfg["layer_num"])+"_best.pt"
    agent.load_model(curr_path, name)
    s,_ = env.reset()
    total_reward = 0
    for i in range(cfg["eval_epsisode"]):
        step = 0 
        while True:
            a = agent.select_action(s)
            s_, r, done, info, _ = env.step(a) 
            s = s_
            #env.render()
            total_reward += r
            step += 1
            if done or step>cfg["n_time_step"]:
                step = 0
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
        q_net_h_layers=agent_dict["hidden_layers"]
    )
    

    print(agent.q_net)
    train(env,agent,cfg)
    #env.close()
    eval(env_name, agent, cfg)

if __name__=="__main__":
    main()
    




