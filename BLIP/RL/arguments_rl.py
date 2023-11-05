import argparse
from datetime import date
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--approach', default='fine-tuning', type=str, required=True,
                    choices=['fine-tuning',
                             'ft-fix', 
                             'ewc', 
                             'blip',
                             'blip_ewc',
                             'blip_spp', 
                             'blip_spp_mask'], 
                    help='(default=%(default)s)')
    parser.add_argument('--experiment', default='atari', type=str, required=True,
                help='Name of the experiment')
    parser.add_argument('--wrapper', default='flat', type=str, required=False,
                choices=['flat', 'img'], 
                help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                choices=['Adam', 'RMSProp'], 
                help='(default=%(default)s)')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=2.5e-4, help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-8,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.0)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 128)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=128,
        help='number of forward steps in PPO/A2C (default: 128 for PPO)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=256,
        help='number of batches for ppo (default: 256)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='eval interval, one eval per n updates (default: 10)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=500000,
        help='number of environment steps to train (default: 500000)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: /logs/)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--ewc-lambda',
        type=float,
        default=5000.0,
        help='lambda for EWC')
    parser.add_argument(
        '--ewc-online',
        default=True,
        help='True == online EWC')
    parser.add_argument(
        '--ewc-epochs',
        type=int,
        default=100,
        help='epochs for EWC')
    parser.add_argument(
        '--num-ewc-steps',
        type=int,
        default=20,
        help='epochs for EWC')
    parser.add_argument(
        '--save-name',
        default=None,
        help='savename for tensorboard')
    parser.add_argument(
        '--date',
        default=date.today(),
        help='date (default: today)')
    parser.add_argument(
        '--task_id',
        default=None,
        type = int,
        help='task_id')
    parser.add_argument(
        '--single-task',
        action='store_true',
        default=False,
        help='whether it is a single task setting')
    parser.add_argument(
        '--F-prior',
        type=float,
        default=1e-16, #5e-18,#1e-15,
        help='prior of Fisher information')
    # for BLIP+EWC and BLIP+SBB approaches
    parser.add_argument('--fisher-term',
        type=str,
        default='f0t',
        help='Use either Fisher t or Fisher 0:t)',
        choices=['f0t','ft'])
    parser.add_argument('--update-bits-rounding',
        type=str,
        default='floor',
        help='Use either floor or ceil)',
        choices=['floor','ceil'])
    parser.add_argument(
        '--spp-lambda',
        type=float,
        default=4.0,
        help='lambda for SPP')
    parser.add_argument(
        '--initial-prune-percent',
        type=float,
        default=30.0,
        help='Initial mask prune percent')
    parser.add_argument(
        '--prune-percent-decay',
        type=float,
        default=0.8,
        help='Prune percent decay')
    parser.add_argument(
        '--use-scheduler',
        action='store_true',
        default=False,
        help='use the linear scheduler in spp')
    parser.add_argument(
        '--prune-higher',
        action='store_true',
        default=False,
        help='Use either lower or higher masking)')

    parser.add_argument('--input-padding', action='store_true', default=False, help='apply no sampling')
    parser.add_argument('--sample', action='store_true', default=False, help='apply no sampling')
    parser.add_argument('--samples', type=int, default=1, help='no of samples to sample')

    # render arguments
    parser.add_argument('--render-ckpt-path',        default='',    type=str, help='path of checkpoint')
    parser.add_argument('--render-task-idx',         default=0,     type=int, help='render task idx')
    parser.add_argument('--num-render-traj',        default=1000,  type=int, help='number of steps rendered')

    # evaluation arguments
    parser.add_argument('--num-eval-episodes',        default=10,  type=int, help='number of episodes for evaluation')
    parser.add_argument('--tasks-sequence',        default=0,  type=int, help='Tasks sequence ID')
    parser.add_argument('--task-state',        default=None,  type=int, help='Load model state at given task')
    parser.add_argument(
        '--metrics-dir',
        default='./metrics/',
        help='directory to save evaluation metrics dataframes (default: ./metrics/)')    

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
