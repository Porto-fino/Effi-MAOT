from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1" # one thread in one process
from setproctitle import setproctitle as ptitle
import torch
# import torch.optim as optim
# from torch.nn import L1Loss
from environment import create_env
from utils import ensure_shared_grads, ensure_shared_grads_param
from model import build_model
from player_util import Agent
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import time
import numpy as np
from queue import Queue

def train(rank, args, shared_model, optimizer, train_modes, n_iters, device, env=None):
    n_steps = 0 #训练过程中的交互次数
    n_iter = 0 #网络更新次数
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent:{}'.format(rank)))
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[0] # 0 i add
    torch.manual_seed(args.seed + rank) #seed = 1 设置CPU生成随机数的种子，方便下次复现实验结果。
    if gpu_id >= 0: # i add
        torch.cuda.manual_seed(args.seed + rank)
    training_mode = args.train_mode #-1
    env_name = args.env #'UnrealUrbanTree-DiscreteColorGoal-v5'

    train_modes.append(training_mode) #将这个训练进程的训练模式改为-1 表示训练，test中会在结束后改为-100 结束训练
    n_iters.append(n_iter) 

    if env == None:###进入
        env = create_env(env_name, args) #建立环境

    params = shared_model.parameters()

    env.seed(args.seed + rank) #env.seed person_id = seed设置目标外观
    player = Agent(None, env, args, None, None, device) #建立智能体
    player.model = build_model(player.env.observation_space, player.env.action_space, args, device).to(device) #建立智能体模型 cuda:0 model上GPU
    
    player.state = player.env.reset() #重置环境 reset返回的state只是相机的观察图片 <class 'numpy.ndarray'>(2, 3, 80, 80) 出现人物
    if 'Unreal' in args.env: #'UnrealMCRoom-DiscreteColorGoal-v5'   True #感觉没啥用 同步player的campose collect_state和env的campose和current_state
        player.cam_pos = player.env.env.env.env.cam_pose ##这个campose哪来的？ -》是reset之后得到的  loc + rot 解释通了
        # 0:[[1108, -3771, 300, 0.0, -11.951354406060238, -9.165048794570714], ]cam_rot[0] roll 翻滚角 cam_rot[1] yaw()：航向，将物体绕Y轴旋转（localRotationY） cam_rot[2] pitch()：俯仰，将物体绕X轴旋转（localRotationX）
        # 1:[2786, -3761, 300, 0.0, -167.72852818618108, -8.902885810824998]
        player.collect_state = player.env.env.env.env.current_states #state是多个相机的观测，原分辨率 2x240x320
        # 0: 1: 
    player.set_cam_info() #player.cam_info 模型推理时posecontroler使用
    player.state = torch.from_numpy(player.state).float() #torch.Size([2, 3, 80, 80]) device(type='cpu')
    player.state = player.state.to(device) #player观测 torch.Size([2, 3, 80, 80])  初始化观测上GPU device(type='cuda', index=0)
    # player.model = player.model.to(device) #player模型 line37已经设置到cuda了

    player.model.train() #设置为训练模式
    reward_sum = torch.zeros(player.num_agents).to(device)
    success_rate_singles = np.zeros(player.num_agents)####### i add this line
    count_eps = 0 #记录交互了几个episode
    # cross_entropy_loss = nn.CrossEntropyLoss() 算gate损失的
    horizon_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置水平误差
    vertical_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置垂直误差
    requester_single_sum = torch.zeros(player.num_agents).to(device)#请求者数量置0
    requester = torch.ones(player.num_agents).to(device)
    link_sum = torch.zeros(player.num_agents).to(device)
    
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict()) #先获取全局模型参数 cuda:0
        # player.model.load_state_dict(shared_model.state_dict()) #获取全局模型参数
        player.update_lstm() #修改player的H_multi torch.zeros(self.num_agents, self.lstm_out) 4,256 why here
        fps_counter = 0
        q = Queue()
        t0 = time.time()
        for step in range(args.num_steps): #20   与环境交互num_steps（5/10/20）步后做一次网络更新
            ifmodelaction = player.action_train()  #采取一次训练动作
            n_steps += 1 #交互一次n_steps就+1  训练过程总交互次数(改为step总次数)
            print('---------环境交互n_steps--------------',n_steps)
            if ifmodelaction != None:
                success_rate_singles += player.success_ids
                reward_sum += player.reward #算的都是2个camera的奖励
                gt_locations = np.array(player.info['gt_locations'])
                horizon_errors += abs(gt_locations[:, 0])
                vertical_errors += abs(gt_locations[:, 1])
                requester_single_sum += player.requester 
                link_sum += player.link_num
            fps_counter +=1# 我自己加的 计算fps用
            if player.done:
                break
        update_steps = len(player.rewards) #一次vision action rewards内容+1 debug 发现没问题
        

        fps = fps_counter / (time.time() - t0) #FPS作用不大

        if player.done:
            print("player done")
            
            #eps结束后 重置q1
            delay = 0.1
            time.sleep(delay)#此轮结束，让可能存在的model进程运行完毕 这句应该也有问题 不光在done后加sleep 应该是在网络每次更新之前(现在也加了)
            if not player.q1.empty():#清空q1
                print("player.q1 本身不空 开始清空q1")
                player.q1.queue.clear()
                if player.q1.empty():
                    print("player.q1 is empty, 清空成功")
                else:#不期望发生
                    print("player.q1清空失败")
            else:
                print("player.q1 本身就是空的")
            
            #重置智能体的flag
            player.flag = True #传输标志置为True（初始状态）
            # horizon_errors = horizon_errors/player.eps_len
            # vertical_errors = vertical_errors/player.eps_len
            mean_error_single = (horizon_errors + vertical_errors)/2
            # horizon_error_mean = horizon_errors.mean()
            # vertical_error_mean = vertical_errors.mean()
            # mean_erroes = (horizon_error_mean + vertical_error_mean)/2

            for i in range(player.num_agents):
                writer.add_scalar('train/reward_'+str(i), reward_sum[i], n_steps)
                writer.add_scalar('train/success_rate_'+str(i), success_rate_singles[i]/player.eps_len, n_steps)
                writer.add_scalar('train/error_rate_'+str(i), mean_error_single[i]/player.eps_len, n_steps)
                
            writer.add_scalar('train/requeseter_num_', requester_single_sum.sum()/player.num_agents/player.eps_len, n_steps)
            writer.add_scalar('train/link_num_', link_sum.sum()/player.num_agents/player.eps_len, n_steps)
            # writer.add_scalar('train/mean_erroes_', mean_erroes, n_steps)
            count_eps += 1 #episode+1
            reward_sum = torch.zeros(player.num_agents).to(device)#奖励之和置0
            # print("----success_rate_singles----",success_rate_singles/player.eps_len)
            success_rate_singles = np.zeros(player.num_agents)#成功率置0
            horizon_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置水平误差
            vertical_errors = np.array([0.0,0.0,0.0,0.0]) #每一轮重置垂直误差
            requester_single_sum = torch.zeros(player.num_agents).to(device) #请求者数量置0
            link_sum = torch.zeros(player.num_agents).to(device)
            player.eps_len = 0
            player.state = player.env.reset() #重置环境
            player.state = torch.from_numpy(player.state).float().to(device) #初始化观测上GPU
            player.set_cam_info()

        R = torch.zeros(player.num_agents, 1).to(device) # 策略网络的输出值

        if not player.done:
            
            state = player.state #最后一次vision_action采集回来的state
            # value_multi, _, _, _, _ = player.model(
            #         (Variable(state, requires_grad=True), player.H_multi), q)
            gt_gate = torch.Tensor(np.array(player.env.env.env.env.gate_ids)).to(device) #gt是ground truth 从环境中看bbox是否大于阈值
            player.model((Variable(state, requires_grad=True), player.H_multi, gt_gate), q, False)#delay_flag = false
            value_multi = q.get()
            value_multi = value_multi.squeeze(1) #我加的 将[2,1,1]->[2,1] 因为其他也都是2维的
            if not q.empty(): ##这是一定的
                print("q不空 开始清空q")
                q.queue.clear()#清空队列
                if q.empty():
                    print("q清空成功")
            else:
                print("q为空")
            
            
            
            for i in range(player.num_agents):
                R[i][0] = value_multi[i].data #交互20次后，当前的状态值 #value_multi.shape torch.Size([4, 1, 1]) R.shape 4,1
        
        ####以下是我写的全局奖励更新方法
        player.values.append(Variable(R).to(device)) #向values中加入当前状态的策略值R[2,1] value_multi.shape torch.Size([2, 1, 1]) player.values[5].shape torch.Size([2, 1])
        policy_loss = torch.zeros(1, 1).to(device) #总和起来看policy loss
        value_loss = torch.zeros(1, 1).to(device) #总和起来看value loss
        entropies = torch.zeros(player.num_agents, 1).to(device)

        w_entropies = torch.Tensor([float(args.entropy)]).to(device) #0.001 torch.Size([4, 1])

        R = Variable(R, requires_grad=True).to(device)#Return
        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(player.rewards))): #倒序range  上次的交互次数''''''''''''''''3 2 1 0
            R = args.gamma * R.sum() + player.rewards[i].sum() #gamma=0.95   player.rewards[i]是2个camera的实际奖励  2X1。R是V(S t+1)
            advantage = R - player.values[i].sum() #优势函数 #2X2X1 player.values[i]是V(St),后来values被我改成torch.Size([2, 1]) n步回报
            value_loss = value_loss + 0.5 * advantage.pow(2) #.pow(2)求平方 #价值函数损失 2x1

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i].sum() + args.gamma * player.values[i + 1].data.sum() - player.values[i].data.sum() #GAE公式中的delta_t torch.Size([2, 1])tensor([[-0.9979],[-0.9990]])
            gae = gae * args.gamma * args.tau + delta_t #torch.Size([2, 1])
            # value_loss = value_loss + gae

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (w_entropies * player.entropies[i].sum())#torch.Size([2, 1])

            entropies += player.entropies[i]

        if update_steps != 0:
            loss = policy_loss / update_steps / player.num_agents + 0.5 * value_loss / update_steps / player.num_agents
            player.model.zero_grad()
            # optimizer.zero_grad()
            print('loss--------',loss)
            delay_update = 0.1
            time.sleep(delay_update)# 更新模型前 让推理的线程运行完毕
            print("slppe done")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 50)
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)

            n_iter += 1 #一次更新(正常是20次)
            # print(n_iter)
            n_iters[rank] = n_iter #记录每个进程训练（更新）了多少inter（一次20步）

            optimizer.step() #优化
            print("更新了一次网络")
        else:
            print("update_steps = 0 无法更新网络")

        player.clear_actions() #清除动作 将前面20步 清除上20步的训练过程保存信息
        
        #save model (copy from test.py)
        if n_iter % 250 == 0 and rank == 0 : #No.1线程保存模型
            # model_dir = os.path.join(args.log_dir, '{0}-model-best-{1}.dat'.format(args.env, n_steps))
            model_dir = os.path.join(args.log_dir, 'model-{0}.pth'.format(n_steps))
            if args.gpu_ids[-1] >= 0:
                with torch.cuda.device(args.gpu_ids[-1]):
                    # state_to_save = player.model.state_dict()
                    # torch.save(state_to_save, model_dir)
                    torch.save(player.model, model_dir)
            else:
                state_to_save = player.model.state_dict()
                # torch.save(state_to_save, model_dir) #保存模型
                torch.save(player.model, model_dir)
                
        if train_modes[rank] == -100: # 在test中也被改变
            env.close()
            break
