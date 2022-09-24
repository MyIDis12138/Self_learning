import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 

import argparse
from datetime import datetime

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 


paser = argparse.ArgumentParser(description="RL with gym")
paser.add_argument('--env', type=str, required=True, help='training env')
paser.add_argument('--algo', type=str, required=True, help='RL algorithm')
paser.add_argument('--ckpt', type=str, default=None, help='load check point before running')
paser.add_argument('--episdo', type=int, default=100, help='the episdos to train')
paser.add_argument('--seed', type=int, default=0, help='seeds for random number')
paser.add_argument('--result_path', type=str, default= curr_path+"/results/" +'/'+curr_time+'/results/', help='path to results')
paser.add_argument('--model_path', type=str, default= curr_path+"/results/" +'/'+curr_time+'/models/', help='path to model')
paser.add_argument('--max_timestep', default=4000000, help='maximum time steps')
