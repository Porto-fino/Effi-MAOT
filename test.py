from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger, check_path
from player_util import Agent
import logging
from tensorboardX import SummaryWriter
import os
from model import build_model
import torch.nn as nn
import time

def test(rank, args, shared_model, train_modes, n_iters, device):
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test Agent:{}'.format(rank)))#log_dir已经具体到环境+时间了
    ptitle('Test Agent: {}'.format(rank)) #修改当前进程名
    torch.manual_seed(args.seed + rank)
    n_iter = 0

    log = {}
    setup_logger('{}_log'.format(args.env),    #测试记录
                 r'{0}/logger'.format(args.log_dir))  #log_dir已经具体到环境+时间了
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k])) #在_log中写内容

    torch.manual_seed(args.seed) # seed = 1 设置CPU生成随机数的种子，方便下次复现实验结果

    env = create_env(args.env, args) #建立环境

    start_time = time.time()
    num_tests = 1
    n_step = 0 
    #              model, env, args, state, cam_info, device
    player = Agent(None, env, args, None, None, device) 
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device).to(device)

    player.state = player.env.reset()
    if 'Unreal' in args.env: ##
        player.cam_pos = player.env.env.env.env.cam_pose #为什么这么长一串 loc + rot 解释通了
        # 0:[-1065, -572, 545, 0.0, 28.19866328421489, -14.428015620557765] cam_rot[0] roll 翻滚角 cam_rot[1] yaw()：航向，将物体绕Y轴旋转（localRotationY） cam_rot[2] pitch()：俯仰，将物体绕X轴旋转（localRotationX）
        # 1:[-779, 606, 277, 0.0, -37.92513575373526, -2.4873929812811113]
        player.collect_state = player.env.env.env.env.current_states #state是相机的观测

    player.set_cam_info()
    player.state = torch.from_numpy(player.state).float().to(device) #把摄像头观测转换为tensor

    player.model.eval()#设置为评估模式
    max_score = -100 #最大分数，下面比较模型用，决定是否保存当前模型
    reward_sum = np.zeros(player.num_agents)
    reward_total_sum = np.zeros(player.num_agents)
    reward_sum_ep = np.zeros(player.num_agents)

    success_rate_sum_ep = np.zeros(player.num_agents)

    fps_counter = 0
    t0 = time.time()
    cross_entropy_loss = nn.CrossEntropyLoss() #定义损失函数

    len_sum = 0
    seed = args.seed

    count_eps = 0
    eps_length = 0
    rate = 0
    rates = [0, 0]
    step_rates = [0, 0]
    mean_rates = [0, 0]

    visible_steps = 0
    while True:
        if player.done:
            count_eps += 1 #评估了多少轮次

            t0 = time.time()
            eps_length = 0

            player.model.load_state_dict(shared_model.state_dict()) #获取当前全局模型参数

        player.action_test() #采取测试动作
        eps_length += 1
        n_step += 1

        fps_counter += 1
        reward_sum_ep += player.reward #累积reward #[0:4] : [1.4354649686848386, 1.7743820905829433, -1.0, -1.0]
        success_rate_sum_ep += player.success_rate #累积成功率 array([1.5, 1.5, 1.5, 1.5])

        gate_ids, gate_probs, gt_gates = [], [], []
        for k1 in range(len(player.rewards)): #1    1个列表 里面4个元素
            for k2 in range(player.num_agents): #4
                _, max_id = torch.max(player.gates[k1][k2], 0)
                gate_probs.append(player.gates[k1][k2])
                gate_ids.append(max_id)
                gt_gates.append(player.gate_gts[k1][k2])

        gate_probs = torch.cat(gate_probs).view(-1, 2).to(device)
        gate_gt_ids = torch.Tensor(gt_gates).view(1, -1).squeeze().long().to(device)
        gate_loss = cross_entropy_loss(gate_probs, gate_gt_ids) #计算开关损失

        visible_steps += sum(np.array(gt_gates).squeeze()) / 4

        gate_ids = np.array([gate_ids[i].cpu().detach().numpy() for i in range(4)])
        gt_gates = np.array([gt_gates[i].cpu().detach().numpy() for i in range(4)])
        one_step_rate = sum(gate_ids == gt_gates) / player.num_agents
        rate += one_step_rate
        for id in range(2):
            right_num = sum(gate_ids[i] == gt_gates[i] == id for i in range(4))
            num = sum(gt_gates[i] == id for i in range(4))
            step_rate = right_num / num if num != 0 else 0
            if step_rate > 0:
                rates[id] += step_rate
                step_rates[id] += 1
                mean_rates[id] = rates[id] / step_rates[id]

        mean_rate = rate / n_step

        if player.done:
            player.state = player.env.reset()
            player.state = torch.from_numpy(player.state).float().to(device)
            player.set_cam_info()

            reward_sum += reward_sum_ep

            len_sum += player.eps_len
            fps = fps_counter / (time.time()-t0)
            n_iter = 0
            for n in n_iters: #所有训练线程中的inter 一个inter代表交互20次
                n_iter += n
            for i in range(player.num_agents):
                writer.add_scalar('test/reward'+str(i), reward_sum_ep[i], n_iter)

            writer.add_scalar('test/fps', fps, n_iter)
            writer.add_scalar('test/eps_len', player.eps_len, n_iter)
            writer.add_scalar('test/unvisible_acc', mean_rates[0], n_iter)
            writer.add_scalar('test/visible_acc', mean_rates[1], n_iter)
            writer.add_scalar('test/mean_acc', mean_rate, n_iter)
            writer.add_scalar('test/gate_loss', gate_loss, n_iter)

            player.eps_len = 0
            fps_counter = 0
            reward_sum_ep = np.zeros(player.num_agents)
            t0 = time.time()
            count_eps += 1
            if count_eps % args.test_eps == 0: #test_eps = 20
                player.max_length = True
            else:
                player.max_length = False

        if player.done and not player.max_length:
            seed += 1
            player.env.seed(seed)
            player.state = player.env.reset()
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float().to(device)

            player.eps_len += 2

        elif player.done and player.max_length:
            ave_reward_sum = reward_sum/args.test_eps
            reward_total_sum += ave_reward_sum
            reward_mean = reward_total_sum / num_tests
            len_mean = len_sum/args.test_eps
            reward_step = reward_sum / len_sum
            log['{}_log'.format(args.env)].info(  #写到logger了
                "Time {0}, ave eps reward {1}, ave eps length {2}, reward mean {3}, reward step {4}".
                format(
                   time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    ave_reward_sum, len_mean, reward_mean, reward_step))

            if ave_reward_sum.mean() >= max_score: #当前模型较好
                print ('save best! in %d iters'%n_step) #保存为最佳模型
                max_score = ave_reward_sum.mean()
                model_dir = os.path.join(args.log_dir, '{0}-gate-all-model-best-{1}.dat'.format(args.env, n_step))
            else: #保存为新模型
                model_dir = os.path.join(args.log_dir, '{0}-new.dat'.format(args.env))

            if args.gpu_ids[-1] >= 0:
                with torch.cuda.device(args.gpu_ids[-1]):
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, model_dir)
            else:
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, model_dir) #保存模型

            num_tests += 1
            reward_sum = 0
            len_sum = 0
            seed += 1
            player.env.seed(seed)

            player.state = player.env.reset()
            if 'Unreal' in args.env:
                player.cam_pos = player.env.env.env.env.cam_pose
                player.collect_state = player.env.env.env.env.current_states
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float().to(device)
            player.input_actions = torch.Tensor(np.zeros((player.num_agents, 9)))

            time.sleep(args.sleep_time)

            if n_iter > args.max_step: #200w
                env.close()
                for id in range(0, args.workers):
                    train_modes[id] = -100 #训练模式改为-100
                break

        player.clear_actions() #清空动作

