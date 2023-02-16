import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from matplotlib.font_manager import FontProperties  # 导入字体模块

def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
        fname='/System/Library/Fonts/STHeiti Light.ttc', size=15) # fname系统字体路径，此处是mac的
    except:
        font = None
    return font

def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):

    plt.figure()
    plt.title(f"{cfg.env_name}env {cfg.algo_name}learning curves")
    plt.xlabel(u'epsiode')
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(f"{cfg.result_path}{tag}_rewards_curve_cn")
    # plt.show()


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(
        f"learning curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}"
    )
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(f"{cfg.result_path}{tag}_rewards_curve")
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    plt.figure()
    plt.title(f"loss curve of {algo}")
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(f"{path}losses_curve")
    plt.show()

def save_results(dic, tag='train', path='./results'):
    ''' 保存奖励
    '''
    for key,value in dic.items():
        np.save(f'{path}{tag}_{key}.npy', value)
    print('Results saved！')
    
# def save_results(rewards, ma_rewards, tag='train', path='./results'):
#     ''' 保存奖励
#     '''
#     np.save(path+'{}_rewards.npy'.format(tag), rewards)
#     np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
#     print('Result saved!')


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    ''' 删除目录下所有空文件夹
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))

def save_args(args):
    # save parameters    
    args_dict = vars(args)
    with open(f'{args.result_path}params.json', 'w') as fp:
        json.dump(args_dict, fp)
    print("Parameters saved!")
def smooth(data, weight=0.9):
    '''_summary_

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                

    return smoothed