from distutils.command.config import config
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path



import gym 
import torch
import datetime
import argparse
from itertools import count

from pg import PolicyGradient
from common.utils import save_args, save_results, make_dir
from common.utils import plot_rewards

class Config:
    def __init__(self) -> None:
        """
        define hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--algo_name',default='PolicyGradient',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_epis',default=3000,type=int,help="episodes of training")
        parser.add_argument('--test_epis',default=20,type=int,help="episodes of testing")
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
        parser.add_argument('--lr',default=0.01,type=float,help="learning rate")
        parser.add_argument('--batch_size',default=8,type=int)
        parser.add_argument('--hidden_dim',default=36,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda")
        parser.add_argument(
            '--result_path',
            default=f"{curr_path}/outputs/{parser.parse_args().env_name}/{curr_time}/results/",
        )
        parser.add_argument(
            '--model_path',
            default=f"{curr_path}/outputs/{parser.parse_args().env_name}/{curr_time}/models/",
        )
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
        args = parser.parse_args()
        self.args = args

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    agent = PolicyGradient(n_states, cfg)
    return env, agent


def train(cfg, env: gym.Env, agent: PolicyGradient):
    print('start training')
    print(f'Env:{cfg.env_name}, Algotithm:{cfg.algo_name}, Device:{cfg.device}')
    state_pool = []
    action_pool = []
    reward_pool = []
    rewards = []
    ma_rewards = []
    steps = []
    for i_epi in range(cfg.train_epis):
            state = env.reset()
            epi_reward = 0
            epi_step = 0

            for _ in count():
                    epi_step += 1
                    action = agent.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    epi_reward += reward
                    if done:
                            reward = 0
                    state_pool.append(state)
                    action_pool.append(float(action))
                    reward_pool.append(reward)
                    state = next_state
                    if done:
                            print(f'Episode:{i_epi+1}/{cfg.train_epis}, Reward:{epi_reward:.2f}')
                            break

            if i_epi > 0 and i_epi % cfg.batch_size ==0: 
                    agent.update(reward_pool, state_pool, action_pool)
                    state_pool = []
                    action_pool = []
                    reward_pool = []               
            steps.append(epi_step)
            rewards.append(epi_reward)
            if ma_rewards:
                    ma_rewards.append(0.9*ma_rewards[-1] + 0.1*epi_reward)
            else:
                    ma_rewards.append(epi_reward)

    print('Finsh traning!')
    env.close()
    return {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}

def test(cfg, env: gym.Env, agent: PolicyGradient):
        print('start test!')
        print(f'Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}')
        rewards = []
        ma_rewards = []
        steps = []
        for i_epi in range(cfg.test_epis):
                state = env.reset()
                epi_reward = 0
                epi_step = 0

                for _ in count():
                        epi_step += 1
                        action = agent.choose_action(state) 
                        next_state, reward, done, _ = env.step(action)
                        epi_reward += reward
                        if done:
                                reward = 0
                        state = next_state
                        if done:
                                print(f'Episode:{i_epi+1}/{cfg.train_epis}, Reward:{epi_reward:.2f}')
                                break
                
                steps.append(epi_step)
                rewards.append(epi_reward)
                if ma_rewards:
                        ma_rewards.append(0.9*ma_rewards[-1] + 0.1*epi_reward)
                else:
                        ma_rewards.append(epi_reward)    
        
        print('Finish test!')
        env.close()
        return {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}          

def main():
    cfg = Config()
    cfg_args = cfg.args
    cfg_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #TRAIN
    env, agent = env_agent_config(cfg_args)
    res_dic = train(cfg_args, env, agent)
    make_dir(cfg_args.result_path, cfg_args.model_path)
    save_args(cfg.args)
    agent.save(path=cfg_args.model_path)
    save_results(res_dic, 
                    path=cfg_args.result_path)
    plot_rewards(res_dic.pop('rewards'), res_dic.pop('ma_rewards'), cfg_args, tag="train")

    #TEST
    env, agent = env_agent_config(cfg_args)
    agent.load(path=cfg_args.model_path)
    res_dic = test(cfg_args, env, agent)
    save_results(res_dic, tag='test',
             path=cfg_args.result_path)
        #plot_rewards(rewards, ma_rewards, cfg, tag="test")

if __name__ == "__main__":
        main()