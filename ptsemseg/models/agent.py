import torch.nn as nn
import torch
from ptsemseg.models.utils import conv2DBatchNormRelu, deconv2DBatchNormRelu, Sparsemax
from ptsemseg.models.backbone import n_segnet_encoder, resnet_encoder, n_segnet_decoder, simple_decoder
import numpy as np


def get_encoder(name): #resnet_encoder
    try:
        return {
            "n_segnet_encoder": n_segnet_encoder,
            "resnet_encoder": resnet_encoder, ###
        }[name]
    except:
        raise ("Encoder {} not available".format(name))

def get_decoder(name): #simple_decoder
    try:
        return {
            "n_segnet_decoder": n_segnet_decoder,
            "simple_decoder": simple_decoder

        }[name]
    except:
        raise ("Decoder {} not available".format(name))


### ============= Modules ============= ###
# class img_encoder(nn.Module):
#     def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1,
#                  enc_backbone='n_segnet_encoder'):
#         super(img_encoder, self).__init__()
#         # feat_chn = 256

#         self.feature_backbone = get_encoder(enc_backbone)(n_classes=n_classes, in_channels=in_channels) #resnet_encoder
#         self.feat_squeezer = feat_squeezer #-1

#         # squeeze the feature map size  压缩特征图 由于feat_squeezer是-1 不压缩
#         if feat_squeezer == 2:  # resolution/2
#             self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=2, padding=1)
#         elif feat_squeezer == 4:  # resolution/4
#             self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=4, padding=1)
#         else:###
#             self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=1, padding=1)

#     def forward(self, inputs):
#         outputs = self.feature_backbone(inputs)
#         outputs = self.squeezer(outputs)

#         return outputs

# class img_decoder(nn.Module):
#     def __init__(self, n_classes=21, in_channels=512, agent_num=5, feat_squeezer=-1, dec_backbone='n_segnet_decoder'):
#         super(img_decoder, self).__init__()

#         self.feat_squeezer = feat_squeezer
#         if feat_squeezer == 2:  # resolution/2
#             self.desqueezer = deconv2DBatchNormRelu(in_channels, in_channels, k_size=3, stride=2, padding=1,
#                                                     output_padding=1)
#             self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=in_channels)

#         elif feat_squeezer == 4:  # resolution/4
#             self.desqueezer1 = deconv2DBatchNormRelu(in_channels, 512, k_size=3, stride=2, padding=1, output_padding=1)
#             self.desqueezer2 = deconv2DBatchNormRelu(512, 512, k_size=3, stride=2, padding=1, output_padding=1)
#             self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=512)
#         else:
#             self.output_decoder = get_decoder(dec_backbone)(n_classes=n_classes, in_channels=in_channels)

#     def forward(self, inputs):
#         if self.feat_squeezer == 2:  # resolution/2
#             inputs = self.desqueezer(inputs)

#         elif self.feat_squeezer == 4:  # resolution/4
#             inputs = self.desqueezer1(inputs)
#             inputs = self.desqueezer2(inputs)

#         outputs = self.output_decoder(inputs)
#         return outputs

### ============= Modules ============= ###
class img_encoder(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, feat_channel=512, feat_squeezer=-1,
                 enc_backbone='n_segnet_encoder'):
        super(img_encoder, self).__init__()
        # feat_chn = 256

        self.feature_backbone = get_encoder(enc_backbone)(n_classes=n_classes, in_channels=in_channels) #resnet_encoder
        self.feat_squeezer = feat_squeezer #-1

        # squeeze the feature map size  压缩特征图 由于feat_squeezer是-1 不压缩
        if feat_squeezer == 2:  # resolution/2
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=2, padding=1)
        elif feat_squeezer == 4:  # resolution/4
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=4, padding=1)
        else:###
            self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.feature_backbone(inputs)
        outputs = self.squeezer(outputs)

        return outputs

class policy_net4(nn.Module):
    def __init__(self, n_classes=21, in_channels=512, input_feat_sz=32, enc_backbone='n_segnet_encoder'):
        super(policy_net4, self).__init__()
        self.in_channels = in_channels #3

        feat_map_sz = input_feat_sz // 4 #8
        self.n_feat = int(256 * feat_map_sz * feat_map_sz) #16384

        self.img_encoder = img_encoder(n_classes=n_classes, in_channels=self.in_channels, enc_backbone=enc_backbone)

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        outputs1 = self.img_encoder(features_map)

        outputs = self.conv1(outputs1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

# class km_generator_back(nn.Module):
#     def __init__(self, out_size=128, input_feat_sz=32):
#         super(km_generator, self).__init__()
#         feat_map_sz = input_feat_sz // 4 # 4 取整除 - 返回商的整数部分（向下取整）
#         self.n_feat = int(256 * feat_map_sz * feat_map_sz) #4096
#         self.fc = nn.Sequential(
#             nn.Linear(self.n_feat, 256), #            
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128), #             
#             nn.ReLU(inplace=True),
#             nn.Linear(128, out_size)) #            

#     def forward(self, features_map):
#         outputs = self.fc(features_map.view(-1, self.n_feat))
#         return outputs
class km_generator(nn.Module):
    def __init__(self, out_size=128):
        super(km_generator, self).__init__()
        feat_map_sz = 4
        self.n_feat = int(16 * feat_map_sz * feat_map_sz) #4096
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs

class linear(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32):
        super(linear, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(256 * feat_map_sz * feat_map_sz)

        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)
        )

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs

class Gate(nn.Module):
    def __init__(self, head_dim):
        super(Gate, self).__init__()
        gate_input_dim = head_dim
        self.feature_dim = 256
        self.gate_fc1 = nn.Linear(gate_input_dim, self.feature_dim)
        # self.gate_fc1.weight.data = norm_col_init(self.gate_fc1.weight.data, 0.1)
        # self.gate_fc1.bias.data.fill_(0)

        self.gate_fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        # self.gate_fc2.weight.data = norm_col_init(self.gate_fc2.weight.data, 0.1)
        # self.gate_fc2.bias.data.fill_(0)

        self.gate_fc3 = nn.Linear(self.feature_dim, 2)
        # self.gate_fc3.weight.data = norm_col_init(self.gate_fc3.weight.data, 0.1)
        # self.gate_fc3.bias.data.fill_(0)

    def forward(self, x):
        feature = torch.relu(self.gate_fc1(x))
        feature = torch.relu(self.gate_fc2(feature))
        gate_prob_value = self.gate_fc3(feature)

        return  gate_prob_value

# <------ Attention ------> #
# MIMO (non warp)
class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1) #列和为1
        self.linear = nn.Linear(query_size, key_size)
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True): #qu torch.Size([2, 6, 32])
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,agent_num,key_size)  把query维度提升到key的维度 可以点乘  torch.Size([2, 6, 1024])

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)



        # generate the       #k:torch.Size([1, 4, 256]) query:torch.Size([1, 4, 256]) query.transpose(2, 1).shape torch.Size([2, 1024, 6])
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # 和我认为的不一样，每一列，是同一个agent的query和其他anget的key(batch(2),6,6)  column: differnt keys and the same query(same agent) ###torch.Size([2, 6, 6])

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,agent_num,agent_num) torch.Size([1, 4, 4]) 列和为1

        attn_shape = attn_orig_softmax.shape   #torch.Size([1, 4, 4])
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]  #1 4 4
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1) #torch.Size([1, 4, 4, 1, 1, 1]) 相当于reshape

        v_exp = torch.unsqueeze(v, 2) #torch.Size([1, 4, 1, 16, 4, 4])
        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1) #torch.Size([1, 4, 4, 16, 4, 4])

        output = attn_orig_softmax_exp * v_exp  # (batch,4,channel,size,size) #torch.Size([1, 4, 4, 16, 4, 4])
        output_sum = output.sum(1)  # (batch,1,channel,size,size) #torch.Size([1, 4, 16, 4, 4]) #4个agent获得融合特征

        return output_sum, attn_orig_softmax

# =======================  Model ========================= 以下类为平行关系
# our model (no warping) #无变形
class MIMOcom(nn.Module):
    def __init__(self, n_classes=9, in_channels=3, feat_channel=512, feat_squeezer=-1, 
                 has_query=True, sparse=False, agent_num=4, shuffle_flag=False, image_size=80,
                  key_size=256, query_size=32):
        super(MIMOcom, self).__init__() #在子类中调用父类的初始化方法

        self.agent_num = agent_num #4
        # self.in_channels = in_channels #3
        # self.shuffle_flag = shuffle_flag #False
        # self.feature_map_channel = 512
        self.key_size = key_size #256  1024
        self.query_size = query_size #32
        self.has_query = has_query #True
        self.sparse = sparse #Flase 这是什么标志


        print('When2com') # our model: detach the learning of values and keys #分离
        
        #Message generator
        if self.has_query:
            # self.query_net = km_generator(out_size=self.query_size, input_feat_sz=image_size / 32)
            self.query_net = km_generator(out_size=self.query_size)
        #以上生成的query_mat key_mat val_mat输入attention_net
        
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size) #32, 1024
        self.gate = Gate(self.key_size)
        # delete List the parameters of each modules

    def activated_select(self, val_mat, prob_action, thres=0.2):
#     torch.Size([2, 6, 512, 16, 16])    torch.Size([2, 6, 6])
        coef_act = torch.mul(prob_action, (prob_action > thres).float()) #torch.mul 每一行对应点相乘 裁剪M为Mbar,但是没有softmax
        attn_shape = coef_act.shape #torch.Size([2, 6, 6])
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1) #torch.Size([2, 6, 6, 1, 1, 1])

        v_exp = torch.unsqueeze(val_mat, 2) #torch.Size([2, 6, 1, 512, 16, 16])
        v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1) #torch.Size([2, 6, 6, 512, 16, 16])

        output = coef_act_exp * v_exp  # (batch,4,channel,size,size) #torch.Size([2, 6, 6, 512, 16, 16])
        feat_act = output.sum(1)  # (batch,1,channel,size,size) #torch.Size([2, 6, 512, 16, 16])

        # compute connect
        count_coef = coef_act.clone() #torch.Size([2, 6, 6])
        ind = np.diag_indices(self.agent_num) #(array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5]))
        count_coef[:, ind[0], ind[1]] = 0 #对角线元素置为0
        num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0]) #/左侧：返回一个包含输入Input中非0元素索引的张量 即发生通信的次数，/右侧为img数据个数 为12个
        return feat_act, coef_act, num_connect
        #feat是Mbar算的融合观测 coef_act是Mbar num_connect是传输次数/智能体数量（12） 也就是平均每个智能体发生多少次通信
    def count_link(self, prob_action, thres=0.25):
#     torch.Size([2, 6, 512, 16, 16])    torch.Size([2, 6, 6])
        # print("M graph",prob_action)
        coef_act = torch.mul(prob_action, (prob_action > thres).float()) #torch.mul 每一行对应点相乘 裁剪M为Mbar,但是没有softmax
        # attn_shape = coef_act.shape #torch.Size([2, 6, 6])
        # bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        # coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1) #torch.Size([2, 6, 6, 1, 1, 1])

        # compute connect
        count_coef = coef_act.clone() #torch.Size([2, 6, 6])
        ind = np.diag_indices(self.agent_num) #(array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5]))
        count_coef[:, ind[0], ind[1]] = 0 #对角线元素置为0
        count_coef = count_coef.view(self.agent_num, self.agent_num)
        # print("graf",count_coef)
        link_num = torch.count_nonzero(count_coef, dim=0)
        # print("link_num",link_num)
        # count_coef = torch.sign(count_coef)#非0元素置1
        # num_connect = torch.nonzero(count_coef).shape[0] / (self.agent_num * count_coef.shape[0]) #/左侧：返回一个包含输入Input中非0元素索引的张量 即发生通信的次数，/右侧为img数据个数 为12个
        return link_num
    
    def agents2batch(self, feats):
        agent_num = feats.shape[1] #feats.shape = torch.Size([2, 6, 512, 16, 16]) agent_num = 6
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0) #torch.Size([12, 512, 16, 16])
        return feat_mat #torch.Size([12, 512, 16, 16])

    def divide_inputs(self, inputs):
        '''
        Divide the input into a list of several images
        '''
        input_list = []
        for i in range(self.agent_num):
            input_list.append(inputs[:, 3 * i:3 * i + 3, :, :])

        return input_list

    def forward(self, inputs, training=True, MO_flag=False , inference='argmax'):
        batch_size, _ = inputs.size()#  2    torch.Size([2, 18, 512, 512])  现在的输入是提取好的特征[4, 256]

        # input_list = self.divide_inputs(inputs) #分成6个智能体各自的观察  len(input_list) = 6   input_list[0].shape = torch.Size([2, 3, 512, 512]) 

        # if self.shared_img_encoder == 'unified':####
        #     # vectorize input list
        #     img_list = []
        #     for i in range(self.agent_num):
        #         img_list.append(input_list[i]) #img_list==input_list True
        #     unified_img_list = torch.cat(tuple(img_list), 0) #torch.Size([12, 3, 512, 512]) 在cuda上

        #     # pass encoder
        #     feat_maps = self.u_encoder(unified_img_list) #unified_encoder torch.Size([12, 512, 16, 16]) cuda:0

        #     # get feat maps for each image
        #     feat_map = {}
        #     feat_list = []
        #     for i in range(self.agent_num): ##   feat_maps: 2,512,16,16
        #         feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)#升一维 torch.Size([2, 1, 512, 16, 16]) 采集出一个anget的2张img的feature
        #         feat_list.append(feat_map[i]) # len(feat_list) = 6 全在GPU
        #     val_mat = torch.cat(tuple(feat_list), 1) # torch.Size([2, 6, 512, 16, 16])
        # else:
        #     raise ValueError('Incorrect encoder')

        
        # pass feature maps through key and query generator
        # query_key_maps = self.query_key_net(unified_img_list) #torch.Size([12, 256, 4, 4]) #feature_map传入query_key_net 生成 query_key_maps，query_key_maps分别传入key和query生成器生成key和query

        # keys = self.key_net(query_key_maps) #torch.Size([12, 1024]) 生成key GPU

        # if self.has_query:
        #     querys = self.query_net(query_key_maps) #torch.Size([12, 32]) 生成query GPU

        # # get key and query
        # key = {}
        # query = {}
        # key_list = []
        # query_list = []

        # for i in range(self.agent_num):
        #     key[i] = torch.unsqueeze(keys[batch_size * i:batch_size * (i + 1)], 1)
        #     key_list.append(key[i]) #len(key_list) = 6 key_list[0].shape torch.Size([2, 1, 1024])
        #     if self.has_query:##
        #         query[i] = torch.unsqueeze(querys[batch_size * i:batch_size * (i + 1)], 1)
        #     else:
        #         query[i] = torch.ones(batch_size, 1, self.query_size).to('cuda')
        #     query_list.append(query[i]) #len(query_list) = 6  query_list[0].shape torch.Size([2, 1, 32])


        # key_mat = torch.cat(tuple(key_list), 1) #torch.Size([2, 6, 1024]) GPU
        # query_mat = torch.cat(tuple(query_list), 1)#torch.Size([2, 6, 32]) GPU

        # if MO_flag:##self.MO_flag = self.cfg['model']['multiple_output'] #True
        #     query_mat = query_mat
        # else:
        #     query_mat = torch.unsqueeze(query_mat[:,0,:],1) #只取第一个智能体的query
        
        
        #让v和q来自HEAD提取到的特征
        val_mat = inputs.view(1, 4, 16, 4, 4) #现在的输入是提取好的特征1, 4, 16, 4, 4 本来是torch.Size([2, 6, 512, 16, 16])
        key_mat = inputs.unsqueeze(0) #现在的输入是提取好的特征1 4 256 本来是torch.Size([2, 6, 1024])
        querys = self.query_net(inputs.view(4, 16, 4, 4)) #torch.Size([4, 32]) 生成query GPU   原来query_net的输入是#torch.Size([12, 256, 4, 4])
        query_mat = querys.unsqueeze(0) #torch.Size([1, 4, 32])
#feat_fuse.shape torch.Size([1, 4, 16, 4, 4])4个智能体的融合观测 prob_action.shape torch.Size([1, 4, 4])
        feat_fuse, prob_action = self.attention_net(query_mat, key_mat, val_mat, sparse=self.sparse)#false
        #output_sum是根据注意力分数，乘以value加权求和得到的4个智能体的融合观测。 attn_orig_softmax就是softmax后的矩阵M
        # gates = self.gate(val_mat)
        
        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1])*0.001 #torch.Size([4, 4]) 生成对角线为1 其余为0的二位数组
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2])) #torch.Size([1, 4, 4])
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).cuda() #torch.Size([1, 4, 4]) 对角线为0.001 其余为0
        prob_action = prob_action + small_bis #torch.Size([1, 4, 4]) 最终生成的softmax后的M矩阵+small bias 有两个 因为batch为2
        link_num = self.count_link(prob_action) #输入v 和M，M做修剪

        if training: ##
            action = torch.argmax(prob_action, dim=1) #torch.Size([2, 6]) tensor([[5, 5, 5, 5, 5, 5],[0, 4, 2, 3, 2, 5]], device='cuda:0') 采集每一列的最大元素的index
            num_connect = self.agent_num - 1 #3 训练的时候 还是全连接 只不过权重不同

            return feat_fuse, prob_action, action, num_connect, link_num #融合观测 M矩阵 根据M矩阵的采样 agent平均连接数
        else: #暂时不修剪M，先训练起来，训练code修改好之前，这部分先折叠隐藏
            if inference == 'activated': #test
                feat_act, connect_mat, num_connect = self.activated_select(val_mat, prob_action) #输入v 和M，M做修剪
#      #feat是Mbar算的融合观测 coef_act是Mbar num_connect是传输次数/智能体数量（12）
                feat_act_mat = self.agents2batch(feat_act)  # (batchsize*agent_num, channel, size, size) #torch.Size([12, 512, 16, 16])
                feat_act_mat = feat_act_mat.detach() #torch.Size([12, 512, 16, 16])

                pred_act = self.decoder(feat_act_mat) #torch.Size([12, 11, 512, 512])预测结果

                action = torch.argmax(connect_mat, dim=1) #torch.Size([2, 6])
#                   Mbar预测结果   M矩阵      Mbar采样动作  平均通信次数
                return pred_act, prob_action, action, num_connect
            else:
                raise ValueError('Incorrect inference mode')