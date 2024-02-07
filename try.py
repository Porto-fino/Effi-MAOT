import torch
import numpy as np
# import torch
import random

# 定义一个 tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# x = torch.Tensor([0.1, 0.5, 0.8, 0.3, 0.7], requires_grad = True).to(device)
# a = torch.ones(4).to(device)
# x = torch.Tensor([1., 0., 0., 1.]).to(device)
# # 定义阈值
# # threshold = 0.5

# # # 利用 torch.where 实现二值化
# # result = torch.where(x > threshold, torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))

# # print(result)
# x = a - x
# print(x)
# y = x.sum()
# print(y)
# coef_act = torch.rand(1,4,4).to(device)
# coef_act = torch.tensor([[[0.6140, 0.0783, 0.3851, 0.5038],
#          [0.3282, 0.3897, 0.5642, 0.9484],
#          [0, 0.7365, 0.2591, 0.5172],
#          [0.9864, 0.7296, 0.1561, 0.7349]]]).to(device)
# print(coef_act)
# count_coef = coef_act.clone() #torch.Size([2, 6, 6])
# print(count_coef)
# ind = np.diag_indices(4) #(array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5]))
# count_coef[:, ind[0], ind[1]] = 0 #对角线元素置为0
# print(count_coef)
# count_coef = count_coef.view(4,4)
# print(count_coef.shape)
# # count_coef = torch.sign(count_coef)

# non_zero_a = torch.count_nonzero(count_coef, dim=0)
# print(non_zero_a)
# # num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0]) #/左侧：返回一个包含输入Input中非0元素索引的张量 即发生通信的次数，/右侧为img数据个数 为12个
# # return feat_act, coef_act, num_connect
# count = count_coef.sum(0)
# print(count)
# gt = torch.tensor([1,0,1,0]).to(device)
# link_num = torch.tensor([1,2,3,4]).to(device)
# num = gt*link_num
# print(num)
a = random.randint(0,3)
print(a)
a = random.randint(0,3)
print(a)
a = random.randint(0,3)
print(a)
a = random.randint(0,3)
print(a)
