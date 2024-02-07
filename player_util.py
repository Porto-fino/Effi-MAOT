from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable
import threading
from queue import Queue

class Agent(object):   #None, env, args, None,  None,     device
    def __init__(self, model, env, args, state, cam_info, device):
        self.model = model # None 建立player时还没有model
        self.env = env
        self.num_agents = len(env.action_space) # 2
        
        self.state = state #None
        self.collect_state = None

        self.cam_info = cam_info #None
        self.cam_pos = None
        self.input_actions = None
        self.hxs = [None for i in range(self.num_agents)]
        self.cxs = [None for i in range(self.num_agents)]
        self.eps_len = 0
        
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.gate_entropies = []
        self.preds = []
        self.done = True #刚创建player是done的
        self.info = None
        self.reward = 0
        self.local_reward = 0
        self.global_reward = 0
        self.device = device
        self.lstm_out = args.lstm_out #256
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0

        self.ds= []
        self.hori_angles = []
        self.verti_angles = []

        self.gt_ds = []
        self.gt_hori_angles = []
        self.gt_verti_angles = []

        # self.gate_probs = []
        # self.gate_entropies = []

        self.images = []
        self.zoom_angle_hs = []
        self.zoom_angle_vs = []
        self.collect_data_step = 0

        # self.last_choose_whos = []
        # self.last_gate_ids = torch.Tensor([1 for i in range(self.num_agents)]).to(self.device)
        self.pre_ids = []
        self.updates = []
        self.gt_ids = []
        # self.last_choose_ids = []
        self.pre_ids = []
        self.single_actions = []
        self.pose_actions = []
        self.cam_poses = []
        self.lstm_features = []
        self.pose_features = []

        self.gate_gts = []
        self.gates = []
        self.time_step = 0
        
        self.q1 = Queue() #训练用队列
        self.q2 = Queue() #测试用队列
        self.flag = True
        self.link_num = torch.zeros(self.num_agents).to(self.device)
    #delete wrap_action() #用不到
    def action_train(self):
        self.gt_gate = torch.Tensor(np.array(self.env.env.env.env.gate_ids)).to(self.device)#各个相机的gate 0/1 tensor([1.,1.,1.,1.])
        if self.q1.empty():#q1空 
            
            action_env_multi = [0, 0, 0, 0] #一定得静止
            
            if self.flag == True: #如果q1空是因为曾经model的计算结果被取走了，那就启动一个新的推理线程
                # self.update_lstm()# 初始化self.H_multi 我加的这句 本来action train中没有 加入后有问题
                cam_action = threading.Thread(target=self.model, args=((self.state, self.H_multi, self.gt_gate), self.q1, True)) #delay_flag 正常推理的时候要delay
                cam_action.start()
                self.flag = False
                print("model开始计算 摄像头静止")
                # value_multi, action_env_multi, self.H_multi , entropy, log_prob = self.model( 
                # (Variable(self.state, requires_grad=True), self.H_multi))
                #输入：
                #self.state torch.Size([2, 3, 80, 80])
                #self.H_multi list length=2  torch.Size([2, 256])
                #输出：
                #value_multi  torch.Size([2, 1, 1]) tensor([[[-0.0195]],[[ 0.0157]]], grad_fn=<ViewBackward0>)
                #action_env_multi  [array(1), array(0)] len()=2
                #self.H_multi len()=2 self.H_multi[0/1].shape torch.Size([2, 256])
                #entropy torch.Size([2, 1])
                #log_prob torch.Size([2, 1]) tensor([[-2.4491],[-2.4668],[-2.3888],[-2.4069]], grad_fn=<CatBackward0>)
            else: #self.flag = False 代表之前的线程还没计算完毕
                print("model未计算完成 摄像头静止")
                #flag 还是 false

            #运算完这个q1为空的情况之后, self.flag一定是False
        
        else : #q1 不空
            print("用vision action")
            self.flag = True
            self.eps_len += 1  #episode长度+1 这一episode增加了
            
            value_multi = self.q1.get() #次
            action_env_multi = self.q1.get()
            self.H_multi = self.q1.get()
            entropy = self.q1.get()
            log_prob = self.q1.get()
            self.requester = self.q1.get()
            self.link_num = self.q1.get()
            if self.q1.empty():
                print("q1true")
            else:
                print('q1false----------------------')             
            [self.Hx, self.Cx] = self.H_multi# self.Hx torch.Size([2, 256]) self.Cx torch.Size([2, 256]) 用模型推理结果交互后，更新隐藏特征
            value_multi = value_multi.squeeze(1) #我加的 将[2,1,1]->[2,1] 因为其他也都是2维的
            self.values.append(value_multi) #
            self.entropies.append(entropy) #算log_probs时用
            self.log_probs.append(log_prob) #算log_probs时用
        
        
        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi) #去与环境交互 #state_multi(2, 3, 80, 80) reward_multi [10.833328582245374, 10.999992854471317] self.done False step中人物体开始移动
        
        
        if self.flag:
            # self.images.append(self.info['states']) #多个相机的观测图像 240.320.3 未resize
            self.state = torch.from_numpy(state_multi).float().to(self.device) #多个相机的观测状态 (2, 3, 80, 80) 
            # self.success_rate = self.info['Success rate']
            self.success_ids = self.info['Success ids']

            if 'Unreal' in self.args.env: ##
                self.cam_pos = self.env.env.env.env.cam_pose #step中会更新cam_pose 
                self.collect_state = self.env.env.env.env.current_states #step中会更新 但是感觉也没有用

            # self.cam_poses.append(self.cam_pos)

            # self.global_reward_mean = sum(reward_multi) / self.num_agents #全局平均奖励
            self.local_reward = torch.tensor(reward_multi).float().to(self.device) #各camera奖励
            self.reward = self.local_reward #tensor([10.8333, 11.0000]) torch.Size([2])
            self.rewards.append(self.reward.unsqueeze(1)) #升一维 [2]->[2,1] rewards[0].shape = torch.Size([2, 1])
            self.set_cam_info()#啥用?
            return self
            
        else:
            return None
        

        

    def action_test_(self):
        # self.gt_gate = torch.Tensor(np.array(self.env.env.env.env.gate_ids)).to(self.device) #gt是ground truth 从环境中看bbox是否大于阈值
        if 'Unreal' in self.args.env: #agent获取环境信息？
            self.cam_pos = self.env.env.env.env.current_cam_pos
            self.collect_state = self.env.env.env.env.current_states
            self.images = self.env.env.env.env.states
            self.target_poses = self.env.env.env.env.current_target_pos

        with torch.no_grad():####
            self.update_lstm()# 初始化self.H_multi
            
            # value_multi, action_env_multi, self.H_multi, entropy, log_prob, R_pred, gate_prob, gate_id, lstm_feature = self.model( #进模型推理
            #     (Variable(self.state), Variable(self.cam_info), self.H_multi, self.last_gate_ids, self.gt_gate)) #state在test.py中设置了，last_gate_ids = [1,1,1,1]
            value_multi, action_env_multi, self.H_multi, entropy, log_prob= self.model( #进模型推理
                (self.state, self.H_multi),q1, delay_flag = True, test=False) 
            self.time_step += 1
        

            # self.gates.append(gate_prob) #tensor([[0.5000, 0.5000],[0.5001, 0.4999],[0.5000, 0.5000],[0.5001, 0.4999]])
            # self.last_gate_ids = gate_id #tensor([0., 0., 1., 0.])

            [self.Hx, self.Cx] = self.H_multi

        

        state_multi, self.reward, self.done, self.info = self.env.step(action_env_multi) # step action_env_multi [2, 1, array(9), 4]
        # print("------------reward-----------------",self.reward)
        self.success_rate = self.info['Success rate']
        self.success_ids = self.info['Success ids']

        # self.gate_probs = gate_prob
        # self.gate_ids = gate_id
        self.actions = action_env_multi
        # self.lstm_features = lstm_feature

        # self.gate_gts.append(self.gt_gate)
        # self.gt_gate = self.info['gate ids']
        self.collect_data_step += 1
        
        if self.args.render:
            if 'gtgate' in self.args.test_type:
                self.env.env.env.env.to_render(self.gt_gate)
            else:
                if 'MCRoom' in self.args.env:
                    self.env.render()
                else:
                    self.env.env.env.env.to_render(gate_id)
                    self.env.render()

        self.state = torch.from_numpy(state_multi).float().to(self.device)

        self.set_cam_info()
        self.eps_len += 1
        self.rewards.append(self.reward)

        return self
    
    def action_test(self):
            
        if self.q2.empty():
            
            action_env_multi = [0,0,0,0]
            if self.flag == True:
                self.flag = False
                self.gt_gate = torch.Tensor(np.array(self.env.env.env.env.gate_ids)).to(self.device) #gt是ground truth 从环境中看bbox是否大于阈值
                with torch.no_grad():
                    self.update_lstm()# 初始化self.H_multi
                    # self.model.eval()
                    # value_multi, action_env_multi, self.H_multi, entropy, log_prob, R_pred, gate_prob, gate_id, lstm_feature = self.model( #进模型推理 gate_id是什么
                    #     (Variable(self.state), Variable(self.cam_info), self.H_multi, self.last_gate_ids, self.gt_gate), self.q2) #state在test.py中设置了，last_gate_ids = [1,1,1,1]
                    cam_action = threading.Thread(target=self.model, args=((self.state, self.H_multi, self.gt_gate), self.q2, True))
                    cam_action.start()
                    self.time_step += 1

                    [self.Hx, self.Cx] = self.H_multi #有用
                    print("模型开始运算.cam静止")
            else:
                print("模型未运算完成.cam静止")
        else:
            print('模型运算完毕, 采用模型action')
            self.flag = True
            value_multi = self.q2.get() #次
            action_env_multi = self.q2.get()
            print("model_action",action_env_multi)
            self.H_multi = self.q2.get()
            entropy = self.q2.get()
            log_prob = self.q2.get()
            if self.q2.empty():
                print("q2true")
            else:
                print('q2false----------------------')  
         

        state_multi, self.reward, self.done, self.info = self.env.step(action_env_multi) # step action_env_multi [2, 1, array(9), 4]
        self.first_inter = False
        # print("------------reward-----------------",self.reward)
        gate_id = [1,1,1,1]
        if self.args.render:
            if self.flag:
                self.env.env.env.env.to_render(gate_id) #运行完这句 弹出画布
            self.env.render() #运行完这句 一行4个cam的窗口
            
        if self.flag:
            self.success_rate = self.info['Success rate'] #sum(cal_target_observed) / self.num_cam
            self.success_ids = self.info['Success ids']
            self.state = torch.from_numpy(state_multi).float().to(self.device)

            self.set_cam_info()
            self.eps_len += 1
            self.rewards.append(self.reward)

            return self
        else:
            return None

    def set_cam_info(self): #这个有啥用吗??实在看不懂
        if self.info: 
            self.cam_pos = self.info['camera poses'] #交互过后就有info了
        self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0) #前把前三列取出（pose），再按列求平均 aarray([ 1947., -3766.,   300.])

        lengths = []
        for i in range(self.num_agents): #[783.75, 68.75, 633.25, 219.25]
            length = np.sqrt(sum(np.array(self.cam_pos[i][:3] - self.coordinate_delta)) ** 2) #对数组每个元素的和返回一个非负平方根
            lengths.append(length) #[844.0, 844.0]
        pose_scale = max(lengths) #找个最大距离 844

        cam = []
        for i in range(self.num_agents):
            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi) #0.2732874642449683 #yaw
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi) #-0.3469874366161386 #pitch
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi) #0.9619324102485346 #yaw
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi) #0.9378697771175704 #pitch

            tmp = np.concatenate([(self.cam_pos[i][:3] - self.coordinate_delta) / pose_scale, #完成多个数组的拼接
                                  np.array([sin_y, cos_y, sin_p, cos_p])])

            cam.append(tmp) #每个摄像头与平均位置的远近 和yaw pitch的三角函数 array([-0.99407583, -0.00592417,  0.        ]) [0:7] : [0.9940758293838863, 0.005924170616113744, 0.0, -0.21254387836576968, -0.977151523444157, -0.15476014657945375, 0.987952072233621]
        self.cam_info = torch.Tensor(np.array(cam)).to(self.device) #torch.Size([2, 7])
        return np.array(cam) #7个内容

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.gate_entropies = []

        self.preds = []
        self.gate_probs = []

        self.pre_ids = []
        self.updates = []

        self.gt_ids = []
        self.pre_ids = []
        # self.last_choose_ids = []
        self.single_actions = []
        self.pose_actions = []

        self.lstm_features = []
        self.pose_features = []
        self.images = []
        self.cam_poses = []

        self.gate_gts = []
        self.gates = []

        return self


    def update_lstm(self):

        if self.done:#如果一轮结束了，那就重置LSTM的数据
            self.cxs = torch.zeros(self.num_agents, self.lstm_out).to(self.device) #torch.Size([2, 256])
            self.hxs = torch.zeros(self.num_agents, self.lstm_out).to(self.device) #variable可以反向传播 torch.Size([2, 256])
            self.H_multi = [self.hxs, self.cxs]
        else:#如果一轮没结束，那就用之前LSTM的数据
            self.Hx , self.Cx = self.Hx.data, self.Cx.data #variable可以反向传播
            self.H_multi = [self.Hx, self.Cx]



