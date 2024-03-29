from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import create_env
from utils import setup_logger
from model import build_model
from player_util import Agent
import gym
import logging
import numpy as np

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--env', default='UnrealGarden-DiscreteColorGoal-v1', metavar='ENV', help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument('--load-vision-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
# parser.add_argument('--load-pose-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='multi-cnn-lstm-discrete', metavar='M', help='Model type to use')
parser.add_argument('--gpu-id', type=int, default=0, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--lstm-out', type=int, default=256, metavar='LO', help='lstm output size')
parser.add_argument('--sleep-time', type=int, default=5, metavar='LO', help='seconds')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose whether to stack observations')
parser.add_argument('--global-model', default='gru', metavar='M', help='Model type to use')
parser.add_argument('--num-episodes', type=int, default=100,metavar='NE', help='how many episodes in evaluation') #测试100轮
parser.add_argument('--test-type', default='modelgate', metavar='M', help='test model type to use:gtgate, modelgate, VisionOnly')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='S', help='random seed (default: 1)')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')

    log = {}
    setup_logger('{0}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{0}_log'.format(args.env)] = logging.getLogger('{0}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{0}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    gpu_id = args.gpu_id

    if gpu_id >= 0:
        torch.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env("{}".format(args.env), args)

    num_tests = 0
    reward_total_sum = 0
    eps_success = 0
    # rewards_his = []
    len_lis = []
    player = Agent(None, env, args, None, None, device)
    # player.model = build_model(
    #     env.observation_space, env.action_space, args, device)
    player.model = torch.load(args.load_vision_model_dir)
    player.gpu_id = gpu_id
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()

    # model_state = player.model.state_dict()
    # if args.load_vision_model_dir is not None:
    #     vision_saved_state = torch.load(
    #         args.load_vision_model_dir,
    #         map_location=lambda storage, loc: storage)
    #     for k, v in model_state.items(): # k和v分别是模型的每一部分的名称和权重
    #         if 'header' in k or 'policy' in k or 'when2com' in k:
    #             model_state[k] = vision_saved_state[k]
    #     player.model.load_state_dict(model_state)
    # vision_saved_state = torch.load(
    #         args.load_vision_model_dir,
    #         map_location=lambda storage, loc: storage)
    # if gpu_id >= 0:
    #     with torch.cuda.device(gpu_id):
    #         player.model.load_state_dict(vision_saved_state)
    # else:
    #         player.model.load_state_dict(vision_saved_state)

    # if args.load_pose_model_dir is not None:

    try:
        player.model.eval() #模型设置为评估模式
        # player.model.train()
        all_horizon_error_mean, all_vertical_error_mean, all_horizon_error_std, all_vertical_error_std = np.zeros(len(env.observation_space)),\
                np.zeros(len(env.observation_space)), np.zeros(len(env.observation_space)), np.zeros(len(env.observation_space))

        all_reward, all_eps_hori_me, all_eps_verti_me, all_eps_hori_st, all_eps_verti_st , all_length , all_success_rate\
            = 0, 0 ,0 , 0, 0, 0, 0
        all_success_rate_single, all_success_rate_single_mean = np.zeros(player.num_agents), np.zeros(player.num_agents)

        for i_episode in range(args.num_episodes):
            print('episode', i_episode)
            if i_episode >= args.num_episodes // 2: #好像没用
                player.env.env.env.reverse = True
            else:
                player.env.env.env.reverse = False

            player.state = player.env.reset()
            if 'Unreal' in args.env:
                player.cam_pos = player.env.env.env.env.cam_pose
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float()
            # player.last_gate_ids = [1 for i in range(player.num_agents)]
            # player.input_actions = torch.Tensor(np.zeros((player.num_agents,11)))
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
            player.eps_len = 0
            reward_sum = np.zeros(len(env.observation_space))
            success_rate_sum = 0
            success_rate_singles = np.zeros(player.num_agents)
            horizon_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置水平误差
            vertical_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置垂直误差

            reward_mean = 0
            eps_step = 0
            
            while True:
                ifmodelaction = player.action_test()
                eps_step += 1
                reward_sum += player.reward
                if ifmodelaction != None:
                    success_rate_sum += player.success_rate
                    success_rate_singles += player.success_ids

                    gt_locations = np.array(player.info['gt_locations'])
                    horizon_errors += abs(gt_locations[:, 0])
                    vertical_errors += abs(gt_locations[:, 1])
                if player.done:
                    num_tests += 1
                    horizon_errors = horizon_errors/player.eps_len
                    vertical_errors = vertical_errors/player.eps_len
                    mean_erroes_single = ((horizon_errors + vertical_errors)/2)
                    horizon_error_mean = horizon_errors.mean()
                    vertical_error_mean = vertical_errors.mean()
                    mean_erroes = (horizon_error_mean + vertical_error_mean)/2
                    agent_reward_mean = np.array(reward_sum).mean()

                    log['{0}_log'.format(args.env)].info(
                        "Hori_mean: {0},  Verti_mean: {1}, reward mean: {2}, Success mean: {3}, Success single: {4}, horizon_errors: {5}, vertical_errors: {6}, mean_errors: {7}".format(
                            horizon_error_mean, vertical_error_mean, agent_reward_mean,
                            success_rate_sum / eps_step, success_rate_singles / eps_step,
                            horizon_errors,vertical_errors,mean_erroes))

                    all_reward += agent_reward_mean
                    all_eps_reward_mean = all_reward / (i_episode + 1)

                    all_success_rate += (success_rate_sum / eps_step)
                    all_success_rate_mean = all_success_rate / (i_episode + 1)

                    all_success_rate_single += (success_rate_singles / eps_step)
                    all_success_rate_single_mean = all_success_rate_single / (i_episode + 1)

                    all_length += eps_step
                    all_eps_length_mean = all_length / (i_episode + 1)

                    all_eps_hori_me += horizon_error_mean
                    all_eps_hori_me_mean = all_eps_hori_me / (i_episode + 1)

                    all_eps_verti_me += vertical_error_mean
                    all_eps_verti_me_mean = all_eps_verti_me / (i_episode + 1)

                    reward_mean = 0
                    success_rate_sum = 0
                    eps_step = 0
                    break

        log['{0}_log'.format(args.env)].info(
            "All Hori_mean, {0},All Verti_mean: {1}, "
            "All reward mean: {2}, All length mean {3}, All success rate mean {4}, Success rate single {5}".
                format(
                all_eps_hori_me_mean, all_eps_verti_me_mean,
                all_eps_reward_mean, all_eps_length_mean, all_success_rate_mean, all_success_rate_single_mean))

    except KeyboardInterrupt:
        print("Shutting down")
        player.env.close()
