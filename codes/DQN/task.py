from os import stat
import gym
import cfg as Arg
import DQN

cfg = Arg.parse_args()


env = gym.make( id = 'CartPole-v1',
                new_step_api=True               
                )
env.reset(seed=1)   

agent = DQN(cfg)

n_states = env.observation_space.shape[0]
n_actions = env.action_space
rewards = []
soomthed_rewards = []
ep_steps = []
for i_episode in range(1, cfg.max_episodes+1):
    state = env.reset()
    ep_reward = 0
    for i_step in range(1, cfg.max_steps+1):
        action = agent.chooose_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_reward+=reward
        


