import sys,os
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
    #"env_name": "ALE/Atlantis-v5",
    "episodes": 500,
    "n_time_step": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay_end": 450,
    "eval_epsisode": 20,
    "layer_num": 3,
    "m_steps": 3
}

agent_dict= {
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "buffer_size": 100000,
        "update_frequency":30,
        "hidden_layers": [128,128],
        "double": True,
        "m_steps": 3
}

class MTstepBuffer:
    def __init__(self):
        self.obs_buffer=[]
        self.obs_buffer_=[]
        self.action_buffer=[]
        self.reward_buffer=[]
        self.done_buffer=[]
    
    def add(self, s, s_, a, r, d):
        self.obs_buffer.append(s)
        self.obs_buffer_.append(s_)
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.done_buffer.append(d)
 
    def clear(self):
        self.obs_buffer=[]
        self.obs_buffer_=[]
        self.action_buffer=[]
        self.reward_buffer=[]
        self.done_buffer=[]


def MT_train(env: gym.Env, agent:DQN, logger:SummaryWriter, cfg):

    s,_ = env.reset()
    epsiode_reward_buffer = []
    done = False
    steps = cfg["m_steps"]
    m_steps_buffer = MTstepBuffer()

    for episode_i in range(cfg["episodes"]):
        episode_reward = 0
        step_i=0
        s,_ = env.reset()

        m_steps_buffer.clear

        epsilon = np.interp(
            episode_i, 
            [0, cfg["epsilon_decay_end"]], 
            [cfg["epsilon_start"], cfg["epsilon_end"]]
            )

        #for step_i in range(cfg["n_time_step"]):
        while True:
            if random.random() > epsilon:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s) 

            s_, r, done, _, info = env.step(a)

            m_steps_buffer.add(s,s_,np.array(a),r,done)

            if step_i>=steps:
                
                reward = sum(
                    agent.gamma**c * r
                    for c, r in enumerate(
                        m_steps_buffer.reward_buffer[step_i - steps : step_i]
                    )
                )

                agent.memory.add(
                    obs=m_steps_buffer.obs_buffer[step_i-steps],
                    obs_=m_steps_buffer.obs_buffer_[step_i],
                    action=m_steps_buffer.action_buffer[step_i-steps],
                    reward=reward,
                    done=m_steps_buffer.done_buffer[step_i]
                )

            s = s_
            episode_reward += r
            agent.update(logger)
            #agent.update_advantage()
            step_i+=1

            if done or step_i>cfg["n_time_step"]:
                break

        logger.add_scalar('episode reward', episode_reward, episode_i)
        logger.add_scalar('epsilon', epsilon, episode_i)
        logger.add_scalar('buffer', agent.memory.size, episode_i)

        if (len(epsiode_reward_buffer)>1 and episode_reward >= max(epsiode_reward_buffer)):
            name = cfg["env_name"]+"_"+str(cfg["layer_num"])+"_best.pt"
            agent.save_model(curr_path, name)

        epsiode_reward_buffer.append(episode_reward)
        print(f"Episode: {episode_i}, reward: {episode_reward}")

    env.close()
    print("Finished!")


def eval(env_name: str, agent:DQN, cfg):
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
    logger = SummaryWriter(cfg["env_name"])

    env = gym.make(env_name)
    agent = DQN(
        env=env,
        batch_size=agent_dict["batch_size"],
        buffer_sizes=agent_dict["buffer_size"],
        gamma=agent_dict["gamma"],
        learning_rate=agent_dict["learning_rate"],
        update_frequency=agent_dict["update_frequency"],
        q_net_h_layers=agent_dict["hidden_layers"],
        double_dqn=agent_dict["double"],
        m_steps=cfg["m_steps"]
    )
    

    print(agent.q_net)
    MT_train(env,agent,logger,cfg)
    #env.close()
    eval(env_name, agent, cfg)

if __name__=="__main__":
    main()