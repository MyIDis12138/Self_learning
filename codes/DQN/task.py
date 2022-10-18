import sys,os
from typing import Dict
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
import random
import numpy as np
from DQN import DQN
from torch.utils.tensorboard import SummaryWriter


cfg = {
    #"env_name": "MountainCar-v0",
    "env_name": "CartPole-v1",
    "episodes": 500,
    "n_time_step": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_end": 300,
    "eval_epsisode": 20,
    "layer_num": 3,
    "evaluation_frequency": 100
}
agent_dict= {
        "batch_size": 32,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "buffer_size": 100000,
        "beta":3,
        "update_frequency":30,
        "hidden_layers": [64,64],
        "ddqn": True,
        "m_steps":1
    }



def train(env: gym.Env, agent:DQN, logger:SummaryWriter, cfg):
    s,_ = env.reset()  
    reward_buffer = []
    max_eval_return = float('-inf')
    eval_turns = 0

    for episode_i in range(1,cfg["episodes"]+1):
        episode_reward = 0
        step_i=0
        s,_ = env.reset()
        done = False
        epsilon = np.interp(episode_i,[0, cfg["epsilon_decay_end"]], [cfg["epsilon_start"], cfg["epsilon_end"]])
        while True:
            if random.random() > epsilon:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s) 
            
            s_, r, done, _, info = env.step(a)
            agent.memory.add(obs=s, action=np.array(a), done=done, reward=r, obs_=s_)

            s = s_
            episode_reward += r
            step_i+=1
            agent.update(logger)
            
            if done or step_i>cfg["n_time_step"]:
                break
        
        logger.add_scalar('episode reward', episode_reward, episode_i)
        logger.add_scalar('epsilon', epsilon, episode_i)
        logger.add_scalar('buffer', agent.memory.size, episode_i)

        print(f"Episode: {episode_i}, reward: {episode_reward}")

        if cfg["evaluation_frequency"]>0 and episode_i%cfg["evaluation_frequency"]==0:
            eval_res = eval(env, agent, cfg)
            eval_turns+=1
            if (eval_res>max_eval_return):
                name = cfg["env_name"]+"_"+str(cfg["layer_num"])+"_best.pt"
                agent.save_model(curr_path, name)
                max_eval_return = eval_res
                logger.add_scalar('eval_res', eval_res, eval_turns)
        
        reward_buffer.append(episode_reward) 
        
    print("Training Finished!")


def eval(env: gym.Env, agent:DQN, cfg: Dict, load_pt:bool = False):
    #env = gym.make(env_name)
    if load_pt:
        name = cfg["env_name"]+"_"+str(cfg["layer_num"])+"_best.pt"
        print(f"load {name}")
        agent.load_model(curr_path, name)
    s,_ = env.reset()
    episode_reward = 0
    eval_reward = 0
    for i in range(cfg["eval_epsisode"]):
        step = 0 
        while True:
            a = agent.select_action(s)
            s_, r, done, info, _ = env.step(a) 
            s = s_
            #env.render()
            episode_reward += r
            step += 1
            if done or step>cfg["n_time_step"]:
                step = 0
                print(f"Evaluation round:{i} total reward: {episode_reward}")
                s,_ = env.reset()
                eval_reward+=episode_reward
                episode_reward = 0 
                break
    #env.close()
    return eval_reward/cfg["eval_epsisode"]

def PER_train(env:gym.Env, agent:DQN, logger:SummaryWriter, cfg):
    s,_ = env.reset()
    reward_buffer = []

    for episode_i in range(cfg["episodes"]):
        delta = 0



def main():
    env_name = cfg["env_name"]#, new_step_api=True) 
    
    writer = SummaryWriter(cfg["env_name"])
    env = gym.make(env_name)
    agent = DQN(
        env=env,
        batch_size=agent_dict["batch_size"],
        buffer_sizes=agent_dict["buffer_size"],
        gamma=agent_dict["gamma"],
        learning_rate=agent_dict["learning_rate"],
        update_frequency=agent_dict["update_frequency"],
        q_net_h_layers=agent_dict["hidden_layers"],
        double_dqn=agent_dict["ddqn"],
        #polyak=0.01
    )
    

    print(agent.q_net)
    train(env,agent,writer,cfg)
    eval(env, agent, cfg, load_pt=True)
    env.close()

if __name__=="__main__":
    main()
    




