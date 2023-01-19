import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from datetime import date

import pickle
import torch
from arguments_rl import get_args

from collections import deque
from rl_module.a2c_ppo_acktr.envs import make_vec_envs
from rl_module.a2c_ppo_acktr.storage import RolloutStorage
from rl_module.evaluation import evaluate
from rl_module.ppo_model import QPolicy, Policy
from stable_baselines3.common.utils import set_random_seed

# Gym MiniGrid specific
import gym
from gym import spaces
from gym.wrappers import Monitor
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from gym.wrappers import Monitor

# define wrapper for CNN Policies
def ImgRGBImgPartialObsWrapper(env):
    return ImgObsWrapper(RGBImgPartialObsWrapper(env))
    
def main():

    # Arguments
    args = get_args()
    conv_experiment = [
        'atari',
    ]

    # Inits
    ########################################################################################################################

    print('Inits...')
    if args.cuda:
        print('[CUDA available]')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('[CUDA unavailable]')
        torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda:0" if args.cuda else "cpu")

    print('Device: ', device)

    task_sequences = [
        (0, 'MiniGrid-DoorKey-6x6-v0'), 
        (1, 'MiniGrid-WallGapS6-v0'), 
        (2, 'MiniGrid-LavaGapS6-v0'),
        (3, 'MiniGrid-RedBlueDoors-6x6-v0')
        ]
    
    task_idx = task_sequences[-1][0]

    save_path = os.path.join(args.save_dir, args.algo)

    # for FlatObsWrapper Minigrid environment
    if args.wrapper == 'flat':
        wrapper_class = FlatObsWrapper
        obs_shape = (2739,)
    # for ImgRGBImgPartialObsWrapper Minigrid environment
    elif args.wrapper == 'img':
        wrapper_class = ImgRGBImgPartialObsWrapper
        obs_shape = (12, 56, 56)

    # Evaluate
    ########################################################################################################################
    print('Experiment: ', args.experiment)
    print('Tasks: ', task_sequences)
    ob_rms = None
    seed_list = [1,2,3]

    print('Evaluating tasks:')

    tot_eval_episode_mean_rewards = []
    tot_eval_episode_std_rewards = []

    for i in range(len(seed_list)):
        print("Seed: ", seed_list[i])
        # Seed
        set_random_seed(seed_list[i])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_list[i])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.approach == 'fine-tuning' or args.approach == 'ft-fix':
            log_name = '{}_{}_{}_{}'.format(args.date, args.experiment, args.approach,seed_list[i])
        elif args.approach == 'ewc' in args.approach:
            log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, seed_list[i], args.ewc_lambda)
        elif args.approach == 'blip':
            log_name = '{}_{}_{}_{}_F_prior_{}'.format(args.date, args.experiment, args.approach, seed_list[i], args.F_prior)

        actor_critic = torch.load(os.path.join(save_path, log_name + '_fullmodel_task_' + str(task_idx) + ".pth"))
        actor_critic.to(device)

        eval_episode_mean_rewards_dict = evaluate(actor_critic, ob_rms, task_sequences, seed_list[i],
                                    args.num_processes, args.log_dir, device, obs_shape, task_idx, args.gamma, wrapper_class=wrapper_class, episodes=args.num_eval_episodes)
        tot_eval_episode_mean_rewards.append(eval_episode_mean_rewards_dict['mean'])
        tot_eval_episode_std_rewards.append(eval_episode_mean_rewards_dict['std'])

    tot_eval_episode_mean_rewards_arr = np.array(tot_eval_episode_mean_rewards, ndmin=2)
    tot_eval_episode_std_rewards_arr = np.array(tot_eval_episode_std_rewards, ndmin=2)

    # Get median reward of 3 evaluations
    tot_eval_episode_mean_rewards_median = np.median(tot_eval_episode_mean_rewards_arr, axis=0)

    # get average of standard deviations
    # tot_eval_episode_mean_rewards_std = np.sqrt(np.mean(np.array(tot_eval_episode_std_rewards, ndmin=2)**2, axis=0))

    print('Final evaluation')
    for i, val_median in enumerate(tot_eval_episode_mean_rewards_median):
        # get index of median value in main array
        index = np.where(tot_eval_episode_mean_rewards_arr==val_median)
        print("Task {}: Evaluation (3 seeds) using {} episodes: median reward {:.5f}, std {:.5f} \n".format(
        i, args.num_eval_episodes, val_median, tot_eval_episode_std_rewards_arr[index[0][0]][index[1][0]]))

if __name__ == '__main__':
    main()
