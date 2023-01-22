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
from rl_module.train_ppo import train_ppo
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter

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

    # Split
    ##########################################################################################################################33
    if args.approach == 'fine-tuning' or args.approach == 'ft-fix':
        log_name = '{}_{}_{}_{}'.format(args.date, args.experiment, args.approach,args.seed)
    elif args.approach == 'ewc' in args.approach:
        log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.ewc_lambda)
    elif args.approach == 'blip':
        log_name = '{}_{}_{}_{}_F_prior_{}'.format(args.date, args.experiment, args.approach, args.seed, args.F_prior)

    if args.experiment in conv_experiment:
        log_name = log_name + '_conv'
    
    save_path = os.path.join(args.save_dir, args.algo)

    ########################################################################################################################
    # Seed
    set_random_seed(args.seed)
    if torch.cuda.is_available():
        print('[CUDA available]')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('[CUDA unavailable]')

    # Inits
    print('Inits...')
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda:0" if args.cuda else "cpu")

    ## Experiment 1
    # taskcla = [(0,7), (1,7), (2,7), (3,7)]
    # task_sequences = [
    #     (0, 'MiniGrid-DoorKey-6x6-v0'), 
    #     (1, 'MiniGrid-WallGapS6-v0'), 
    #     (2, 'MiniGrid-LavaGapS6-v0'),
    #     (3, 'MiniGrid-RedBlueDoors-6x6-v0')       
    #     ]
    ## Experiment 2
    # task_sequences = [
    #     (0, 'MiniGrid-RedBlueDoors-6x6-v0'), 
    #     (1, 'MiniGrid-LavaGapS6-v0'), 
    #     (2, 'MiniGrid-DoorKey-6x6-v0'),
    #     (3, 'MiniGrid-WallGapS6-v0')
    #     ]
    ## Experiment 3
    taskcla = [(0,7), (1,7), (2,7), (3,7), (4,7)]
    # task_sequences = [
    #     (0, 'MiniGrid-DoorKey-6x6-v0'), 
    #     (1, 'MiniGrid-WallGapS6-v0'), 
    #     (2, 'MiniGrid-LavaGapS6-v0'),
    #     (3, 'MiniGrid-RedBlueDoors-6x6-v0'),
    #     (4, 'MiniGrid-Empty-Random-6x6-v0')        
    #     ]
    ## Experiment 4
    task_sequences = [
        (0, 'MiniGrid-LavaGapS6-v0'),
        (1, 'MiniGrid-DoorKey-6x6-v0'), 
        (2, 'MiniGrid-Empty-Random-6x6-v0') 
        (3, 'MiniGrid-RedBlueDoors-6x6-v0'),
        (4, 'MiniGrid-WallGapS6-v0')
        ]

    # for FlatObsWrapper Minigrid environment
    if args.wrapper == 'flat':
        wrapper_class = FlatObsWrapper
        obs_shape = (2739,)
    # for ImgRGBImgPartialObsWrapper Minigrid environment
    elif args.wrapper == 'img':
        wrapper_class = ImgRGBImgPartialObsWrapper
        obs_shape = (12, 56, 56)

    if args.approach == 'blip':
        from rl_module.ppo_model import QPolicy
        print('using fisher prior of: ', args.F_prior)
        actor_critic = QPolicy(obs_shape,
            taskcla,
            base_kwargs={'F_prior': args.F_prior, 'recurrent': args.recurrent_policy}).to(device)
    else:
        from rl_module.ppo_model import Policy
        actor_critic = Policy(obs_shape,
            taskcla,
            base_kwargs={'recurrent': args.recurrent_policy}).to(device)

    # Args -- Approach
    if args.approach == 'fine-tuning' or args.approach == 'ft-fix':
        from rl_module.ppo import PPO as approach

        agent = approach(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=True,
                optimizer=args.optimizer)
    elif args.approach == 'ewc':
        from rl_module.ppo_ewc import PPO_EWC as approach

        agent = approach(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            ewc_lambda= args.ewc_lambda,
            online = args.ewc_online,
            optimizer=args.optimizer)

    elif args.approach == 'blip':
        from rl_module.ppo_blip import PPO_BLIP as approach

        agent = approach(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            optimizer=args.optimizer)

    ########################################################################################################################

    print('Device: ', device)
    print('Approach: ', args.approach)
    print('Date: ', args.date)
    print('Experiment: ', log_name)
    print('Length task sequence: ', len(task_sequences))
    print('Tasks: ', task_sequences, '\n')

    tr_reward_arr = []
    te_reward_arr = {}

    for _type in (['mean', 'max', 'min', 'std']):
        te_reward_arr[_type] = {}
        for idx in range(len(taskcla)):
            te_reward_arr[_type]['task' + str(idx)] = []

    # Define Tensorboard loggers

    tb_logger_train = {
        'header':["update", "num_timesteps", "FPS", "mean_reward", "median_reward", "min_reward", "max_reward", "std", "entropy", "value_loss", "action_loss"],
        'writer':SummaryWriter(os.path.join("./result_data/", log_name, "train"))
    }

    tb_logger_eval = {
        'header':["update", "num_timesteps", "mean_reward", "min_reward", "max_reward", "std"],
        'writer':[]
    }

    for idx in range(len(taskcla)):
        tb_logger_eval['writer'].append(SummaryWriter(os.path.join("./result_data/", log_name, "eval", "task_"+str(idx))))

    prev_total_num_steps = 0    
    
    for task_idx,env_name in task_sequences:
        print('Training task '+str(task_idx)+': '+env_name)
        # renew optimizer
        agent.renew_optimizer()

        # FlatObsWrapper for MiniGrid
        envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False, wrapper_class=wrapper_class)
        obs = envs.reset()

        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                        obs_shape, envs.action_space,
                                        actor_critic.recurrent_hidden_state_size)

        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)
        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

        current_total_num_steps = train_ppo(actor_critic, agent, rollouts, task_idx, env_name, task_sequences, envs,  obs_shape, args, episode_rewards, tr_reward_arr, te_reward_arr, tb_logger_train, tb_logger_eval, num_updates, log_name, device, prev_total_num_steps, wrapper_class=wrapper_class)

        prev_total_num_steps = current_total_num_steps

        # post-processing
        if args.approach == 'fine-tuning':
            if args.single_task == True:
                envs.close()
                break
            else:
                envs.close()
        elif args.approach == 'ft-fix':
            # fix the backbone
            for param in actor_critic.features.parameters():
                param.requires_grad = False
            if args.single_task == True:
                envs.close()
                break
            else:
                envs.close()
        elif args.approach == 'ewc':
            agent.update_fisher(rollouts, task_idx)
            envs.close()
        elif args.approach == 'blip':
            agent.ng_post_processing(rollouts, task_idx)
            # save the model here so that bit allocation is saved
            #save_path = os.path.join(args.save_dir, args.algo)
            torch.save(actor_critic.state_dict(),
                os.path.join(save_path, log_name + '_task_' + str(task_idx) + ".pth"))
            envs.close()

    # Save full-model (for ewc evaluation)
    torch.save(actor_critic,
                os.path.join(save_path, log_name + '_fullmodel_task_' + str(task_idx) + ".pth"))

    ########################################################################################################################

if __name__ == '__main__':
    main()
