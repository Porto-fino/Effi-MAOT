from __future__ import division
import torch
import torch.nn as nn
# import torch.nn.init as init
# from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init
import perception
import numpy as np
# import math
import time
from ptsemseg.models import MIMOcom
import random

def build_model(obs_space, action_space, args, device):
    name = args.model #'multi-cnn-lstm-discrete'

    model = A3C_MULTI(args, obs_space, action_space, args.lstm_out, name, args.stack_frames, device)
                    #Box(3,80,80)*2 #Discreate(11)*2 256 'multi-cnn-lstm-discrete'   1          cpu
    model.train() # 设置为训练模式
    return model


#    multi-cnn-lstm-discrete torch.Size([1, 11])
def sample_action(action_type, mu_multi, test=False, gpu_id=-1): #对单个摄像头进行动作采样 
    if 'discrete' in action_type: #multi-cnn-lstm-discrete
        logit = mu_multi #[1,11]一个摄像头的概率分布tensor([[ 0.0195,  0.0131,  0.0627, -0.0020,  0.0441, -0.0524,  0.0118,  0.0149,0.0342,  0.0307,  0.0014]], grad_fn=<SelectBackward>) #未归一化过的概率
        prob = F.softmax(logit, dim=1) #按行归一化  计算动作概率 tensor([[0.0912, 0.0906, 0.0952, 0.0892, 0.0934, 0.0848, 0.0905, 0.0908, 0.0925,0.0922, 0.0895]], grad_fn=<SoftmaxBackward>)
        log_prob = F.log_softmax(logit, dim=1) #softmax后做ln tensor([[-2.4142, -2.4078, -2.3982, -2.3446, -2.3852, -2.3323, -2.4718, -2.4064, -2.3500, -2.4267, -2.4491]], grad_fn=<LogSoftmaxBackward0>)
        entropy = -(log_prob * prob).sum(1) #sum1 按行求和 tensor([2.3970], grad_fn=<NegBackward0>)
        if test:
            action = prob.max(1)[1].data #测试阶段按照概率最大采动作 
        else:### pose中，验证阶段也用这个 其实不对
            action = prob.multinomial(1).data #input可以看成一个权重张量，每一个元素的值代表其在该行中的权重。采样 tensor([[10]])
            log_prob = log_prob.gather(1, Variable(action)) #tensor([[-2.4491]], grad_fn=<GatherBackward0>) 取所选动作的log_prob
        action_env_multi = np.squeeze(action.cpu().numpy()) #array(10)

    return action_env_multi, entropy, log_prob


class HEAD(torch.nn.Module):#(3,80,80)   256   'multi-cnn-lstm-discrete'    1
    def __init__(self, obs_space, lstm_out=256, head_name='cnn_lstm',  stack_frames=1):

        super(HEAD, self).__init__()
        self.head_name = head_name
        if 'cnn' in head_name:
            self.encoder = perception.CNN_net(obs_space, stack_frames) #(3,80,80) 1
        feature_dim = self.encoder.outdim  #1024
        self.head_cnn_dim = self.encoder.outdim #1024
        if 'lstm' in head_name:
            self.lstm = nn.LSTMCell(feature_dim, lstm_out) #(1024,256)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = lstm_out #256
            
        self.head_dim = feature_dim #256

    def forward(self, inputs):
        X, (Hx, Cx) = inputs #X torch.Size([2, 3, 80, 80])  Hx torch.Size([2, 256])  Cx torch.Size([2, 256])
        feature_cnn = self.encoder(X) #feature_cnn [4,1024] device(type='cuda', index=0)

        if 'lstm' in self.head_name:
            Hx, Cx = self.lstm(feature_cnn, (Hx, Cx)) #Hidden_state 短期记忆，也是lstm的output，Cell_state 长期记忆，hidden_size(lstm_out)也是这两个state的维度，因此也是输出的唯独 device(type='cuda', index=0)
            feature = Hx #torch.Size([4, 256])

        return feature, (Hx, Cx) #feature torch.Size([4, 64, 4, 4]) Hx:torch.Size([2, 256])  Cx:torch.Size([2, 256]) torch.Size([2, 256])

class Policy(torch.nn.Module): #256  9         256        'multi-cnn-lstm-discrete'   1
    def __init__(self, outdim, action_space, lstm_out=128, head_name='cnn_lstm',  stack_frames=1):
        super(Policy, self).__init__()
        self.head_name = head_name
        if 'lstm' in self.head_name: ##
            feature_dim = lstm_out   #256
        else:
            feature_dim = outdim

        #  create actor
        if 'discrete' in head_name:  #true
            num_outputs = action_space.n #9

        self.actor_linear1 = nn.Linear(feature_dim*2, feature_dim) ##更改连接方式后修改这里
        self.actor_linear = nn.Linear(feature_dim, num_outputs)
        self.actor_linear1.weight.data = norm_col_init(self.actor_linear1.weight.data, 0.1) #设置权重
        self.actor_linear1.bias.data.fill_(0) #设置bias
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1) #设置权重
        self.actor_linear.bias.data.fill_(0) #设置bias
        

        # create critic
        self.critic_linear1 = nn.Linear(feature_dim*2, feature_dim) ##更改连接方式后修改这里
        self.critic_linear = nn.Linear(feature_dim, 1)
        self.critic_linear1.weight.data = norm_col_init(self.critic_linear1.weight.data, 0.1) #设置权重
        self.critic_linear1.bias.data.fill_(0) #设置bias
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1) #设置权重
        self.critic_linear.bias.data.fill_(0) #设置bias

    def forward(self, feature): #torch.Size([2, 1, 512])
        
        feature1= self.critic_linear1(feature) #torch.Size([2, 1, 256]) 
        value = self.critic_linear(feature1) # value.shape torch.Size([2, 1, 1])  feature.shape torch.Size([4, 1, 256])
        if 'discrete' in self.head_name:##
            feature2 = self.actor_linear1(feature) #torch.Size([2, 1, 256])
            mu = self.actor_linear(feature2) # mu.shape torch.Size([2, 1, 9]) 代表各动作的概率

        return value, mu #状态值和动作概率分布 value torch.Size([4, 1, 1]) mu torch.Size([4, 1, 9])


class A3C_MULTI(torch.nn.Module):#Box(3,80,80)*4  #Discreate(11)*4  #256  'multi-cnn-lstm-discrete'   1    device(type='cuda', index=0) 
    def __init__(self, args, obs_space, action_spaces, lstm_out=256, head_name='cnn_lstm',  stack_frames=1, device=None):
        super(A3C_MULTI, self).__init__()
        self.num_agents = len(obs_space)  # 智能体数量=观测长度  4
        # self.global_name = args.global_model # 'gru' 对于pose控制器来说 是考虑全局pose 所以是global_model

        obs_shapes = [obs_space[i].shape for i in range(self.num_agents)] #Obs_shapes = [(3,80,80) * 4]
        self.head_name = head_name  #'multi-cnn-lstm-discrete'
                            #(3,80,80)      256   'multi-cnn-lstm-discrete'  1
        self.header = HEAD(obs_shapes[0], lstm_out, head_name, stack_frames) # 'multi-cnn-lstm-discrete' CNN和LSTM  head_dim=256
        self.policy = Policy(self.header.head_dim, action_spaces[0], lstm_out, head_name, stack_frames)
                                        #256              9             256  'multi-cnn-lstm-discrete'  1
        self.device = device   # 原来训练用cpu
        self.Hx = torch.zeros(1, lstm_out) #hn是分线
        self.Cx = torch.zeros(1, lstm_out) #hc是主线
        
        #建立when2com网络
        self.when2com = MIMOcom(
                      n_classes=9, #11
                    #   in_channels=in_channels, #3 RGB
                      has_query=True, #True
                      sparse=False, #False
                      agent_num=self.num_agents, #4  原来是6
                      image_size=args.input_size, #80 原来是512
                      query_size=32, #32
                      key_size=256#1024
                    )
        
    def forward(self, inputs, q1, delay_flag, test=False): #test=False从未被设置 一定是false 但是感觉eval中应该得是true
        s_time = time.time()
        requester_ = torch.ones(self.num_agents).to(self.device)
        #多个camera的观测 pos信息 LSTM信息
        states, H_states , gt_gate= inputs #state torch.Size([2, 3, 80, 80]) H_state：2 X torch.Size([2, 256])数据都在cuda:0
                        #gt_gate tensor([1.,1.,1.,1.])
        feature, (Hx, Cx) = self.header((states, H_states)) #header输出 4camera feature,Hx,Cx的shape都是torch.Size([4, 256]) 都在cuda:0上 feature=Hx
        ###header提取到特征了，这里加when2com工作 做通信剪枝
        feature_fuse , M, action ,num_connect , link_num_single= self.when2com(feature, training = True, MO_flag = True)
        #融合观测 M矩阵 根据M矩阵的采样 agent平均连接数 #feat_fuse.shape torch.Size([1, 4, 16, 4, 4])4个智能体的融合观测
        requester = (requester_ - gt_gate)
        link_num_single = link_num_single * requester
        feature_fuse = feature_fuse.view(4, 256) #cuda0
        # print("训练过程,gtgate",gt_gate)
        for i in range(self.num_agents): #gt_gate是门控标准答案
            if gt_gate[i] == 0:
                # print("观测不好")
                feature_fuse[i] = feature_fuse[i]
            elif gt_gate[i] == 1:
                feature_fuse[i] = feature[i]
                # print("观测良好")
            else:
                print("门控错误")
        feature_share = torch.zeros(4,512).to(self.device)
        #our method 4 512
        feature_share[0] = torch.cat((feature[0],feature_fuse[0]),dim = 0)
        feature_share[1] = torch.cat((feature[1],feature_fuse[1]),dim = 0)
        feature_share[2] = torch.cat((feature[2],feature_fuse[2]),dim = 0)
        feature_share[3] = torch.cat((feature[3],feature_fuse[3]),dim = 0)
        
        #random---- 4 512
        # for i in range(self.num_agents):
        #     a = random.randint(0,3)
        #     feature_fuse[i] = feature[a]
        #     feature_share[i] = torch.cat((feature[i],feature_fuse[i]),dim = 0)
        #random----
       
        if delay_flag:
            delay = 0.1002
            time.sleep(delay)
        
        # #以下为全连接 需要修改# below catall 
        # feature_share = torch.zeros(4,1024).to(self.device) #这个在cpu上
        # feature_share[0] = torch.cat((feature[0],feature[1],feature[2],feature[3]),dim = 0)
        # feature_share[1] = torch.cat((feature[1],feature[0],feature[2],feature[3]),dim = 0)
        # feature_share[2] = torch.cat((feature[2],feature[0],feature[1],feature[3]),dim = 0)
        # feature_share[3] = torch.cat((feature[3],feature[0],feature[1],feature[2]),dim = 0)
        # above catall
        Hiden_states = (Hx, Cx) #lstm输出 (图片的序列编码) 2* torch.Size([2, 256])
        # values, single_mus = self.policy(feature_fuse.unsqueeze(1)) # (4, 1, 256) critic的value 和 actor的动作 都是多个相机的 feature为什么增加一维????? input(batch,*,inputsize) 均在cuda0
        #values.shape  torch.Size([2, 1, 1]) single_mus.shape torch.Size([2, 1, 11])
        
        values, single_mus = self.policy(feature_share.unsqueeze(1)) #imput:torch.Size([4, 1, 512])
        # print("single_mus",single_mus)
        vision_actions = []
        entropies = []
        log_probs = []

        for i in range(self.num_agents): #2 把推理结果取出
            vision_action, entropy, log_prob = sample_action(self.head_name, single_mus[i], test) #'multi-cnn-lstm-discrete'  ith camera 动作概率  test 在train的时候是false
            vision_actions.append(vision_action)
            entropies.append(entropy)
            log_probs.append(log_prob)
        # print("vision_actions",vision_actions)
        log_probs = torch.cat(log_probs) #torch.Size([2, 1])  在给定维度上对输入的张量序列seq 进行连接操作
        entropies = torch.cat(entropies) # len(entropies) 2
        entropies = entropies.unsqueeze(1) #torch.Size([2, 1]) 这里操作的好,shape变成[2,1]
        # tensor([[2.3970],
        # [2.3971], grad_fn=<UnsqueezeBackward0>)

        
        final_actions = [] #4个摄像头的action
        
        for i in range(self.num_agents):
            final_actions.append(vision_actions[i])
        
        # self.flag = True
        q1.put(values)
        q1.put(final_actions)
        q1.put(Hiden_states)
        q1.put(entropies)
        q1.put(log_probs)
        q1.put(requester)
        q1.put(link_num_single)
        # return values, final_actions, Hiden_states, entropies, log_probs
#             [2,1,1]      len()=2    Hiden_states[0/1] shape = torch.Size([2, 256])  entrop torch.Size([2, 1]) log_prob torch.Size([2, 1])
        if delay_flag:
            print("单独线程中model运算完成!!!!!!!!")
        else:
            print("模型更新过程运算完毕")
        # print("4cam model Time cost", e_time - s_time)