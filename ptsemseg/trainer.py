import os
import time
import torch
from tqdm import tqdm

from ptsemseg.metrics import runningScore, averageMeter

from ptsemseg.utils import convert_state_dict

#delete other trainer

class Trainer_MIMOcom(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer #tensorboard的writer
        self.logger = logger #log
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer #Adam
        self.scheduler = scheduler #constantLR 保持不变的

        self.n_classes = 11 #这是语义分割的类别
        self.loss_fn = loss_fn


        self.running_metrics_val = runningScore(self.n_classes)   #验证时的metrics（只有验证时使用）
        self.device = device #cuda
        self.MO_flag = self.cfg['model']['multiple_output'] #True

        # some datasets have no labels for communication
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label'] #mimo
        else:
            self.if_commun_label = 'None'

    def train(self):
        # load model
        print('LearnMIMOCom_Trainer')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None: #self.cfg["training"]["resume"] = ‘None’   resume:重新开始，继续；再次出现
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:####
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()#Computes and stores the average and current value
        time_meter = averageMeter()#Computes and stores the average and current value

        best_iou = -100.0
        i = start_iter #0
        flag = True

        # training
        while i <= self.cfg["training"]["train_iters"] and flag: #200000轮 我改成100000轮了
            for data_list in self.trainloader: #batch = 2
                i += 1
                start_ts = time.time()

                if self.if_commun_label != 'None':#### len(labels_list)=6  labels_list.shape torch.Size([2, 512, 512]) 是mask（label）  commun_label.shape torch.Size([2, 2, 6])
                    images_list, labels_list, commun_label = data_list #len(images_list) = 6 images_list[0].shape = torch.Size([2, 3, 512, 512]) batch channel size 是图片
                # else:
                #     images_list, labels_list = data_list
                images = torch.cat(tuple(images_list), dim=1) #torch.Size([2, 18, 512, 512]) 6个agent的img

                if self.MO_flag:  # multiple output #####模型是多输出的 即6个智能体的推理结果一起输出
                    labels = torch.cat(tuple(labels_list), dim=0) #torch.Size([12, 512, 512]) 6个agent的mask
                # else:  # single output
                #     labels = labels_list[0]

                self.scheduler.step() #constantLR 保持不变的  暂时不知道这个是在干什么 可以debug看看
                self.model.train()  # matters for batchnorm/dropout

                images = images.to(self.device) #images上GPU
                labels = labels.to(self.device) #labels上GPU
                if self.if_commun_label != 'None':######
                    commun_label = commun_label.to(self.device) #torch.Size([2, 2, 6]) #通信labels上GPU


                # image loss
                self.optimizer.zero_grad()
                outputs, log_action, action_argmax, _ = self.model(images, training=True, MO_flag=self.MO_flag) #output torch.Size([12, 11, 512, 512]) action_argmax torch.Size([2, 6])
                #pred, prob_action, action, num_connect
                loss = self.loss_fn(input=outputs, target=labels) #仅做下游任务监督，更新整个网络  #tensor(2.4223, device='cuda:0', grad_fn=<NllLossBackward0>)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)
                # Process display on screen
                if (i + 1) % self.cfg["training"]["print_interval"] == 0: #50 轮打印一次 这里再+1是为何？？？？？
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i + 1,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )
                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                    time_meter.reset()

                ###  Validation
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]: # 1000轮验证一次 或者200000次的时候封顶了
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, data_list in tqdm(enumerate(self.valloader)): #tqdm进度条库

                            if self.if_commun_label != 'None':####  len(labels_val_list) = 6 labels_val_list[0].shape = torch.Size([2, 512, 512])  commin_label.shape torch.Size([2, 2, 6])
                                images_val_list, labels_val_list, commun_label = data_list #len(images_val_list) = 6 images_val_list[0].shape = torch.Size([2, 3, 512, 512]) 
                                commun_label = commun_label.to(self.device)
                            else:
                                images_val_list, labels_val_list = data_list
                            images_val = torch.cat(tuple(images_val_list), dim=1) #torch.Size([2, 18, 512, 512])

                            if self.MO_flag:  # obtain multiple ground-truth ###
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)
                            else:  # only select one view gt mask
                                labels_val = labels_val_list[0]

                            labels_val = labels_val.to(self.device) #torch.Size([12, 512, 512])
                            images_val = images_val.to(self.device) #torch.Size([2, 18, 512, 512])
                            gt = labels_val.data.cpu().numpy() #(12, 512, 512) 真值2x6x512x512

                            # image loss
                            outputs, _, action_argmax, _ = self.model(images_val, training=True, MO_flag=self.MO_flag) #outputs torch.Size([12, 11, 512, 512])  action_argmax.shape = torch.Size([2, 6]) 验证阶trinning为什么也是true
                            val_loss = self.loss_fn(input=outputs, target=labels_val)
                            pred = outputs.data.max(1)[1].cpu().numpy() #(12, 512, 512) outputs.data是12x11x512x512的 max(1)[1]是取第二维的最大值的索引，就是该像素点的分类结果。预测值2x6x512x512

                            # compute action accuracy
                            if self.if_commun_label != 'None':#mimo
                                self.running_metrics_val.update_div(self.if_commun_label, gt, pred, commun_label) #commun_label torch.Size([2, 2, 6])按图片进行batch
                                self.running_metrics_val.update_selection(self.if_commun_label, commun_label, action_argmax)#计算when2com和who2com正确的智能体个数

                            self.running_metrics_val.update(gt, pred) #所有12个图片的像素分类混淆矩阵
                            val_loss_meter.update(val_loss.item()) #平均损失

                    if self.if_commun_label != 'None': ###通信动作的准确性
                        when2com_acc, who2com_acc = self.running_metrics_val.get_selection_accuracy()
                        print('Validation when2com accuracy:{}'.format(when2com_acc))
                        print('Validation who2com accuracy:{}'.format(who2com_acc))
                        self.writer.add_scalar("val_metrics/when_com_accuacy", when2com_acc, i) #是否需要通信  
                        self.writer.add_scalar("val_metrics/who_com_accuracy", who2com_acc, i) #与谁通信

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    print('Normal')
                    score, class_iou = self.running_metrics_val.get_only_normal_scores() #when2com通信预测正确的agent的指标
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print('Noise')
                    score, class_iou = self.running_metrics_val.get_only_noise_scores() #when2com通信预测错误的agent的指标
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print("Overall")
                    score, class_iou = self.running_metrics_val.get_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    # store the best model
                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)
                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path #返回模型保存路径

    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)

    def evaluate(self, testloader,inference_mode='activated'): # "val_split"

        running_metrics = runningScore(self.n_classes)#11x11

        # Setup Model
        self.model.eval()
        self.model.to(self.device)


        for i, data_list in enumerate(testloader):#取出值并生成索引
            if self.if_commun_label:## len(labels_list)  labels_list[0].shape torch.Size([2, 512, 512]) commun_label.shape = torch.Size([2, 2, 6])
                images_list, labels_list, commun_label = data_list #len(images_list) = 6 images_list[0].shape = torch.Size([2, 3, 512, 512])
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1) #images_list[0].shape torch.Size([2, 3, 512, 512])
            #images.shape = torch.Size([2, 18, 512, 512])

            # multiple output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)
            outputs, _, action_argmax, bandW = self.model(images, training=False, MO_flag=self.MO_flag, inference=inference_mode)


            pred = outputs.data.max(1)[1].cpu().numpy() #(12, 512, 512)
            gt = labels.numpy() #(12, 512, 512)

            # measurement results
            running_metrics.update(gt, pred)
            running_metrics.update_bandW(bandW) #把每次的 通信次数/agent数 （即平均每个agent通信几次）

            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)
                running_metrics.update_selection(self.if_commun_label, commun_label, action_argmax)

        if self.if_commun_label: #mimo 进入
            when2com_acc, who2com_acc = running_metrics.get_selection_accuracy()
            print('Validation when2com accuracy:{}'.format(when2com_acc))
            print('Validation who2com accuracy:{}'.format(who2com_acc))
        else:
            when2com_acc = 0
            who2com_acc = 0


        avg_bandW = running_metrics.get_avg_bandW()
        print('Bandwidth: ' + str(avg_bandW)) #计算整个验证过程中，平均每个智能体通信次数


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou