# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes)) #(11, 11)
        self.confusion_matrix_pos = np.zeros((n_classes, n_classes)) #(11, 11)
        self.confusion_matrix_neg = np.zeros((n_classes, n_classes)) #(11, 11)
        self.total_agent = 0
        self.correct_when2com = 0
        self.correct_who2com = 0
        self.total_bandW = 0 
        self.count = 0

    def update_bandW(self, bandW):
        self.total_bandW += bandW
        self.count += 1.0 

    def update_selection(self, if_commun_label, commun_label, action_argmax):
        if if_commun_label == 'when2com':
            action_argmax = torch.squeeze(action_argmax)
            commun_label = commun_label + 1  # -1,0,1,2,3 ->0, 1, 2 ,3 ,4


            self.total_agent += commun_label.size(0)
            when_to_commu_label = (commun_label == 0)

            if action_argmax.dim() == 2:
                predict_link = (action_argmax > 0.2).nonzero()
                link_num = predict_link.shape[0]
                when2com_pred = torch.zeros(commun_label.size(0), dtype=torch.int8)

                for row_idx in range(link_num):
                    sample_idx = predict_link[row_idx,:][0]
                    link_idx = predict_link[row_idx,:][1]
                    if link_idx == commun_label[sample_idx]:
                        self.correct_who2com = self.correct_who2com +1
                    if link_idx != 0:
                        when2com_pred[sample_idx] = True
                when2com_pred = when2com_pred.cuda()
                self.correct_when2com += (when2com_pred == when_to_commu_label).sum().item()
            elif action_argmax.dim() == 1:
                # Learn when to communicate accuracy
                when_to_commu_pred = (action_argmax == 0)
                self.correct_when2com += (when_to_commu_pred == when_to_commu_label).sum().item()

                # Learn who to communicate accuracy
                self.correct_who2com += (action_argmax == commun_label).sum().item()
            else:
                assert commun_label.shape == action_argmax.shape, "Shape of selection labels are different."
        elif if_commun_label == 'mimo':

            # commun_label = commun_label.cpu()
            self.total_agent += commun_label[:,0,:].shape[0]*commun_label[:,0,:].shape[1] #每次+12
            # when2com
            id_tensor = torch.arange(action_argmax.shape[1]).repeat(action_argmax.shape[0], 1)#tensor([[0, 1, 2, 3, 4, 5],[0, 1, 2, 3, 4, 5]])
            when_to_commu_pred = (action_argmax.cpu() != id_tensor)#tensor([[ True,  True,  True,  True,  True, False],[ True,  True,  True, False,  True,  True]]) TRUE代表需要通信 False代表不需要通信
            when_to_commu_label = commun_label[:,0,:].type(torch.ByteTensor)
            self.correct_when2com += (when_to_commu_pred == when_to_commu_label).sum().item() #是否需要通信的判断，即when2com

            # who2com (gpu)   who *(need com)
            gt_action = commun_label[:,1,:] * commun_label[:,0,:] + id_tensor.cuda()*(1 - commun_label[:,0,:])#+左边的部分 代表的是需要通信且与谁通信。右边是不需要通信的智能体 但是要有自己的观测

            self.correct_who2com += (action_argmax == gt_action).sum().item()#action max意思是谁和自己的相关性系数最大，需要通信的那部分agent，与别人的相关性系数最大；不需要通信的智能体，与自己的相关性系数最大

    def update_div(self, if_commun_label, label_trues, label_preds, commun_label):
        # import pdb;pdb.set_trace()
        if if_commun_label == 'when2com':
            commun_label = commun_label.cpu().numpy()
            when2comlab = (commun_label == -1) # -1 ---> noraml # other --> noist need com
        elif if_commun_label == 'mimo':###
            commun_label = commun_label.cpu().numpy()[:, 0, :] #把两个batch的第一行取出来 是代表哪个智能体需要通信 array([[1, 0, 0, 1, 0, 0],[0, 0, 0, 1, 1, 0]])
            when2comlab = (commun_label == 0) # 0 --> normal # 1 ---> noisy need com  array([[False,  True,  True, False,  True,  True],[ True,  True,  True, False, False,  True]])分别是img1和img2对应的6个智能体是否需要通信
            when2comlab = when2comlab.transpose(1, 0) #[batch of agent1, batch of agent2 ] #转置
            when2comlab = when2comlab.flatten() #展平array([False,  True,  True,  True,  True,  True, False, False,  True, False,  True,  True])

        # import pdb; pdb.set_trace()
        pos_idx = (when2comlab == True).nonzero() #0: array([ 1,  2,  3,  4,  5,  8, 10, 11]) #信息足够 不需通信的索引
        neg_idx = (when2comlab == False).nonzero() #(array([0, 6, 7, 9]),) #信息不足 需通信的索引

        label_trues_pos = label_trues[pos_idx] #(8, 512, 512) #信息足够 不需通信的索引的mask
        label_preds_pos = label_preds[pos_idx] #(8, 512, 512) #信息足够 不需通信的索引的预测
        if label_trues_pos.shape[0] != 0 :
            for lt, lp in zip(label_trues_pos, label_preds_pos): #将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                self.confusion_matrix_pos += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)#11x11 混淆矩阵

        label_trues_neg = label_trues[neg_idx] #(4, 512, 512) #信息不足 需要通信的索引的mask
        label_preds_neg = label_preds[neg_idx] #(4, 512, 512) #信息不足 需要通信的索引的预测
        if label_trues_neg.shape[0] != 0:
            for lt, lp in zip(label_trues_neg, label_preds_neg): #lt (512, 512) lp (512, 512)
                self.confusion_matrix_neg += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)#11x11 混淆矩阵



    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2 #最少生成n_class 平方个
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_avg_bandW(self):
        return self.total_bandW/self.count

    def get_only_noise_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix_neg
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )


    def get_only_normal_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc #带权重交并比
        """
        hist = self.confusion_matrix_pos
        acc = np.diag(hist).sum() / hist.sum() #对角线之和/所有像素和 求整体准确性
        acc_cls = np.diag(hist) / hist.sum(axis=1) #每一行的对角线元素/每一行的合（实际为某一类） 求每个类的准确度 1是行 0是列
        acc_cls = np.nanmean(acc_cls)#np.nanmean()中：nan取值为0且取均值时忽略它 平均准确度
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) #6个类的IOU
        mean_iu = np.nanmean(iu) #平均 IoU
        freq = hist.sum(axis=1) / hist.sum() #实际为每个类别的像素/所有像素 每个类别出现的频率
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum() #* 按位乘 并求和    #带权重交并比
        cls_iu = dict(zip(range(self.n_classes), iu)) #每个类别的iou

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc, #带权重交并比
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )


    def get_selection_accuracy(self):
        when_com_accuacy = self.correct_when2com / self.total_agent * 100
        who_com_accuracy = self.correct_who2com / self.total_agent * 100

        return  when_com_accuacy, who_com_accuracy




    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_agent = 0
        self.correct_when2com = 0
        self.correct_who2com = 0
        self.total_bandW = 0 
        self.count = 0


    def print_score(self,n_classes, score, class_iou):
        metric_string = ""
        class_string = ""

        for i in range(n_classes):
            # print(i, class_iou[i])
            metric_string = metric_string + "  " + str(i)
            class_string = class_string + " " + str(round(class_iou[i] * 100, 2))

        for k, v in score.items():
            metric_string = metric_string + "  " + str(k)
            class_string = class_string + " " + str(round(v * 100, 2))
            # print(k, v)
        print(metric_string)
        print(class_string)


class averageMeter(object): #meter 仪表
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self): #重置清零
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val #当前值
        self.sum += val * n #和
        self.count += n #累计次数
        self.avg = self.sum / self.count #平均值
