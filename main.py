from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1" # one thread in one process
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import build_model
from train import train
from test import test
from shared_optim import SharedAdam
import time
from datetime import datetime


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)') #fix 0.001
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor for rewards (default: 0.95)') #fix 0.99
parser.add_argument('--tau', type=float, default=0.95, metavar='T', help='parameter for GAE (default: 1.00)')             #fix 0.95
parser.add_argument('--entropy', type=float, default=0.01, metavar='T', help='parameter for entropy (default: 0.001)')    #fix 0.01
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, default=1, metavar='W', help='how many training processes to use (default: 6)')
parser.add_argument('--testers', type=int, default=1, metavar='W', help='how many test processes to collect data (default: 1)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 20)') #fix 10
parser.add_argument('--test-eps', type=int, default=20, metavar='M', help='maximum length of an episode (default: 20)')
parser.add_argument('--env', default='UnrealMCRoom-DiscreteColorGoal-v5', metavar='ENV', help='environment to train on')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-vision-model-dir', default=None, metavar='LMD', help='folder to load trained vision models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='multi-cnn-lstm-discrete', metavar='M', help='viison model type to use')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--render', dest='render', action='store_true', help='render test ')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use an optimizer without shared statistics.')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='training mode')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='choose number of observations to stack')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--lstm-out', type=int, default=256, metavar='LO', help='lstm output size')
parser.add_argument('--sleep-time', type=int, default=5, metavar='LO', help='seconds')
parser.add_argument('--step-size', type=int, default=10000, metavar='LO', help='step size for lr schedule')
parser.add_argument('--max-step', type=int, default=2000000, metavar='LO', help='max learning steps')


if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1: #如果在code中写0,args.gpu_ids就是一个int 如果在训练命令中写，就是一个list[]
        args.gpu_ids = [-1]
        device = torch.device('cpu')####
    else:
        # device = torch.device('cuda:' + str(args.gpu_ids[0])) # cuda:0
        device = torch.device('cuda:' + str(args.gpu_ids[-1])) #device(type='cuda', index=0)
    env = create_env(args.env, args)

    shared_model = build_model( #建立共享模型
        env.observation_space, env.action_space, args, device)  # obs_space Box(3,80,80) *2  action_space Discrete(11)*2
    
    print('model', shared_model)                                #args参数   device：cpu
   
    # delete 加载模型代码

    params = shared_model.parameters() #保存共享模型参数
   
    shared_model.share_memory() #share_memory，它允许数据处于一种特殊的状态，可以在不需要拷贝的情况下，任何进程都可以直接使用该数据。
    
    
    #optimizer代码由以下3行代替
    print ('share memory')
    optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
    optimizer.share_memory() #Adam
    
    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time) #log_dir在这里被修改了'logs/UnrealUrbanTree-DiscreteColorGoal-v1/Nov21_18-56'
    # env.close()

    processes = []
    manager = mp.Manager()
    train_modes = manager.list() #用于共享
    n_iters = manager.list() #用于共享

    # for rank in range(0, args.testers): #1
    #     p = mp.Process(target=test, args=(rank, args, shared_model, train_modes, n_iters, device))
    #     p.start()
    #     processes.append(p)
    #     time.sleep(args.sleep_time)

    for rank in range(0, args.workers): # ranke = 0-5 表示第几个
        p = mp.Process(target=train, args=(
            rank, args, shared_model ,optimizer,train_modes, n_iters, device))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)
    for p in processes:
        # time.sleep(args.sleep_time) i think this is useless.
        p.join()