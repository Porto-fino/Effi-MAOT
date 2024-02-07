# from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from utils import norm_col_init, weights_init, weights_init_mlp


class CNN_net(nn.Module):# (3,80,80)
    def __init__(self, obs_shape, stack_frames):
        super(CNN_net, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2) #参数1 in_channel = obs_shape = 3 参数2 out_channel = 32
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu') #返回给定非线性函数的推荐增益值 relu是给sqrt(2) 2开方
        self.conv1.weight.data.mul_(relu_gain) #x和y点对点相乘，其中x.mul_(y)是in-place操作，会把相乘的结果存储到x中
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))# (1,3,80,80) 随机生成一个输入
        out = self.forward(dummy_state) #先推理了一下
        self.outdim = out.size(-1) #out.shape = torch.Size([1,1024]) out.size(-1) = 1024
        self.apply(weights_init) #初始化权重
        self.train() #设为训练模式

    def forward(self, x):#[2,3,80,80]
        x = F.relu(self.maxp1(self.conv1(x))) #选动作的时候，处理完之后torch.Size([2, 32, 40, 40])
        x = F.relu(self.maxp2(self.conv2(x))) #选动作的时候，处理完之后torch.Size([2, 32, 19, 19])
        x = F.relu(self.maxp3(self.conv3(x))) #选动作的时候，处理完之后torch.Size([2, 64, 9, 9])
        x = F.relu(self.maxp4(self.conv4(x)))#torch.Size([1, 64, 4, 4]) #选动作的时候，处理完后torch.Size([4, 64, 4, 4])
        x = x.view(x.shape[0], -1) #x.shape[0]=1 说明转换成一行，展平。 选动作的时候，处理完后，torch.Size([4, 1024])
        return x #torch.Size([1, 1024]) #选动作的时候，处理完后 [2, 1024] device(type='cuda', index=0)