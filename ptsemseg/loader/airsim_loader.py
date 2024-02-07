import matplotlib
matplotlib.use('Agg')

import os
import torch
import numpy as np
import glob
import cv2
import copy
from random import shuffle
import random
from torch.utils import data
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt




def label_region_n_compute_distance(i,path_tuple):
    begin = path_tuple[0] #(0, 0)
    end = path_tuple[1] #(16, -74)

    # computer distance
    distance = ((begin[0]-end[0])**2 +(begin[1]-end[1])**2)**(0.5) #75.70997开方


    # label region
    if begin[0] <= -400 or end[0]< -400:
        region = 'suburban'
    else:
        if begin[1] >= 300 or end[1] >=300:
            region = 'shopping'
        else:
            region = 'skyscraper'


    # update tuple
    path_tuple = (i,)+path_tuple + (distance, region,)#(0, (0, 0), (16, -74), 75.7099729229908, 'skyscraper')

    return path_tuple




class airsimLoader(data.Dataset):

    # segmentation decoding colors 
    name2color = {"person":      [[135, 169, 180]],
                  "sidewalk":    [[242, 107, 146]], 
                  "road":        [[156,198,23],[43,79,150]],
                  "sky":         [[209,247,202]],
                  "pole":        [[249,79,73],[72,137,21],[45,157,177],[67,266,253],[206,190,59]],
                  "building":    [[161,171,27],[61,212,54],[151,161,26]],
                  "car":         [[153,108,6]],
                  "bus":         [[190,225,64]],
                  "truck":       [[112,105,191]],
                  "vegetation":  [[29,26,199],[234,21,250],[145,71,201],[247,200,111]]
                 }


    name2id = {"person":      1,
               "sidewalk":    2, 
               "road":        3,
               "sky":         4,
               "pole":        5,
               "building":    6,
               "car":         7,
               "bus":         8,
               "truck":       9,
               "vegetation":  10 }


    id2name = {i:name for name,i in name2id.items()}

    splits = ['train', 'val', 'test']
    image_modes = ['scene', 'segmentation_decoded']

    weathers = ['async_rotate_fog_000_clear']

    # list of nodes on the maps (this needs for loading from the folder)  此处边上点的y和dataset中的文件名的y为相反数？？why？？！！
    all_edges = [ #107条边
        ((0, 0), (16, -74)),
        ((16, -74), (-86, -78)),
        ((-86, -78), (-94, -58)),
        ((-94, -58), (-94, 24)),
        ((-94, 24), (-143, 24)),
        ((-143, 24), (-219, 24)),
        ((-219, 24), (-219, -68)),
        ((-219, -68), (-214, -127)),
        ((-214, -127), (-336, -132)),
        ((-336, -132), (-335, -180)),
        ((-335, -180), (-216, -205)),
        ((-216, -205), (-226, -241)),
        ((-226, -241), (-240, -252)),
        ((-240, -252), (-440, -260)),
        ((-440, -260), (-483, -253)),
        ((-483, -253), (-494, -223)),
        ((-494, -223), (-493, -127)),
        ((-493, -127), (-441, -129)),
        ((-441, -129), (-443, -222)),
        ((-443, -222), (-339, -221)),
        ((-339, -221), (-335, -180)),
        ((-219, 24), (-248, 24)),
        ((-248, 24), (-302, 24)),
        ((-302, 24), (-337, 24)),
        ((-337, 24), (-593, 25)),
        ((-593, 25), (-597, -128)),
        ((-597, -128), (-597, -220)),
        ((-597, -220), (-748, -222)),
        ((-748, -222), (-744, -128)),
        ((-744, -128), (-746, 24)),
        ((-744, -128), (-597, -128)),
        ((-593, 25), (-746, 24)),
        ((-746, 24), (-832, 27)),
        ((-832, 27), (-804, 176)),
        ((-804, 176), (-747, 178)),
        ((-747, 178), (-745, 103)),
        ((-745, 103), (-696, 104)),
        ((-696, 104), (-596, 102)),
        ((-596, 102), (-599, 177)),
        ((-599, 177), (-747, 178)),
        ((-599, 177), (-597, 253)),
        ((-596, 102), (-593, 25)),
        ((-337, 24), (-338, 172)),
        ((-337, 172), (-332, 251)),
        ((-337, 172), (-221, 172)),
        ((-221, 172), (-221, 264)),
        ((-221, 172), (-219, 90)),
        ((-219, 90), (-219, 24)),
        ((-221, 172), (-148, 172)),
        ((-148, 172), (-130, 172)),
        ((-130, 172), (-57, 172)),
        ((-57, 172), (-57, 194)),
        ((20, 192), (20, 92)),
        ((20, 92), (21, 76)),
        ((21, 76), (66, 22)),
        ((66, 22), (123, 28)),
        ((123, 28), (123, 106)),
        ((123, 106), (123, 135)),
        ((123, 135), (176, 135)),
        ((176, 135), (176, 179)),
        ((176, 179), (210, 180)),
        ((210, 180), (210, 107)),
        ((210, 107), (216, 26)),
        ((216, 26), (118, 21)),
        ((118, 21), (118, 2)),
        ((118, 2), (100, -62)),
        ((100, -62), (89, -70)),
        ((89, -70), (62, -76)),
        ((62, -76), (28, -76)),
        ((28, -76), (16, -74)),
        ((16, -74), (14, -17)),
        ((-494, -223), (-597, -220)),
        ((-597, -128), (-493, -127)),
        ((-493, -127), (-493, 25)),
        ((-336, -132), (-337, 24)),
        ((14, -17), (66, 22)),
        ((-597, 253), (-443, 253)),
        ((-443, 253), (-332, 251)),
        ((-332, 251), (-221, 264)),
        ((-221, 264), (-211, 493)),
        ((-211, 493), (-129, 493)),
        ((-129, 493), (23, 493)),
        ((23, 493), (20, 274)),
        ((176, 274), (176, 348)),
        ((176, 348), (180, 493)),
        ((180, 493), (175, 660)),
        ((175, 660), (23, 646)),
        ((23, 646), (-128, 646)),
        ((-128, 646), (-134, 795)),
        ((-134, 795), (-130, 871)),
        ((-130, 871), (20, 872)),
        ((175, 872), (175, 795)),
        ((252, 799), (175, 795)),
        ((175, 795), (23, 798)),
        ((23, 798), (-134, 795)),
        ((-134, 795), (-128, 676)),
        ((-128, 676), (-129, 493)),
        ((23, 493), (23, 646)),
        ((23, 646), (23, 798)),
        ((23, 798), (20, 872)),
        ((-338, 172), (-332, 251)),
        ((-57, 255), (20, 255)),
        ((-57, 194), (20, 192)),
        ((20, 255), (20, 274)),
        ((20, 274), (176, 267)),
        ((23, 493), (180, 493)),
        ((176, 267), (176, 348))]
    split_subdirs = {}
    ignore_index = 0
    mean_rgb = {"airsim": [103.939, 116.779, 123.68],}  

    def __init__(
        self,
        root, #作为数据加载器，root为数据集目录
        split="train",
        subsplit=None,
        is_transform=False,
        img_size=(512, 512),
        augmentations=None,
        img_norm=True,
        commun_label='None',
        version="airsim",
        target_view="target"

    ):

        # dataloader parameters 
        self.dataset_div = self.divide_region_n_train_val_test() #数据集分组
        self.split_subdirs = self.generate_image_path(self.dataset_div)# train val test的文件路径
        self.commun_label = commun_label #mimo
        self.root = root #'dataset/airsim-mrms-noise-data'
        self.split = split #根据不同的loader 分别是train val test
        self.is_transform = is_transform #True
        self.augmentations = augmentations #None
        self.img_norm = img_norm #True
        self.n_classes = 11  #什么意思 应该是语义分割分类有11类
        self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))#(512, 512)
        self.mean = np.array(self.mean_rgb[version]) #airsim 的平均RGB

        # Set the target view; first element of list is target view 
        self.cam_pos = self.get_cam_pos(target_view) #['agent1', 'agent2', 'agent3', 'agent4', 'agent5', 'agent6']

        # load the communication label
        if self.commun_label != 'None':
            comm_label = self.read_selection_label(self.commun_label)

        # Pre-define the empty list for the images 
        self.imgs = {s:{c:{image_mode:[] for image_mode in self.image_modes} for c in self.cam_pos} for s in self.splits}
        self.com_label = {s:[] for s in self.splits}

        k = 0
        for split in self.splits: # [train, val, test]
            for subdir in self.split_subdirs[split]: # [trajectory ]  train val test的文件路径 例如 '-337_-172__-221_-172*'

                file_list = sorted(glob.glob(os.path.join(root, 'scene', 'async_rotate_fog_000_clear',subdir,self.cam_pos[0],'*.png'),recursive=True)) #该轨迹上agent1的观测(scene)

                for file_path in file_list: #filelist 00 : 'dataset/airsim-mrms-noise-data/scene/async_rotate_fog_000_clear/-337_-172__-221_-172/agent1/frame001727.png'
                    ext = file_path.replace(root+"/scene/",'') #'async_rotate_fog_000_clear/-337_-172__-221_-172/agent1/frame001727.png'
                    file_name = ext.split("/")[-1] #'frame001727.png'
                    path_dir = ext.split("/")[1] #'-337_-172__-221_-172'

                    # Check if a image file exists in all views and all modalities
                    list_of_all_cams_n_modal = [os.path.exists(os.path.join(root,modal,'async_rotate_fog_000_clear',path_dir, cam,file_name)) for modal in self.image_modes for cam in self.cam_pos] 
                    #12个True 就是2个modal[scene segmentation_decoded] x 6个agent[1-6]
                    if all(list_of_all_cams_n_modal):
                        k += 1
                        # Add the file path to the self.imgs 
                        for comb_modal in self.image_modes: #将这个图片路径添加到self.imags的6个agent中
                            for comb_cam in self.cam_pos:
                                file_path = os.path.join(root,comb_modal,'async_rotate_fog_000_clear', path_dir,comb_cam,file_name)
                                self.imgs[split][comb_cam][comb_modal].append(file_path)

                        if self.commun_label != 'None': # Load the communication label 
                            self.com_label[split].append(comm_label[path_dir+'/'+file_name])

        if not self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]]:#不进入
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.root)
            )
        print("Found %d %s images" % (len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]]), self.split))

    # <---- Functions for conversion of paths ----> 
    def tuple_to_folder_name(self, path_tuple):
        start = path_tuple[1]
        end = path_tuple[2]
        path=str(start[0])+'_'+str(-start[1])+'__'+str(end[0])+'_'+str(-end[1])+'*' #此处对y添加-
        return path
    def generate_image_path(self, dataset_div):

        # Merge across regions
        train_path_list = []
        val_path_list = []
        test_path_list = []
        for region in ['skyscraper','suburban','shopping']:
            for train_one_path in dataset_div['train'][region][1]:
                train_path_list.append(self.tuple_to_folder_name(train_one_path)) #该函数给y坐标添加上了-
            
            for val_one_path in dataset_div['val'][region][1]:
                val_path_list.append(self.tuple_to_folder_name(val_one_path))
            
            for test_one_path in dataset_div['test'][region][1]:
                test_path_list.append(self.tuple_to_folder_name(test_one_path))


        split_subdirs = {}
        split_subdirs['train'] = train_path_list
        split_subdirs['val'] = val_path_list
        split_subdirs['test'] = test_path_list

        return split_subdirs
    def divide_region_n_train_val_test(self):

        region_dict = {'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]}  #区域字典
        test_ratio = 0.25
        val_ratio = 0.25

        dataset_div= {'train':{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]},
                      'val'  :{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]},
                      'test' :{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]}}

        process_edges = []
        # label and compute distance
        for i, path in enumerate(self.all_edges): #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            process_edges.append(label_region_n_compute_distance(i,path))#(0, (0, 0), (16, -74), 75.7099729229908, 'skyscraper')

            region_dict[process_edges[i][4]][1].append(process_edges[i])
            region_dict[process_edges[i][4]][0] = region_dict[process_edges[i][4]][0] + process_edges[i][3] #累计每个tuple的距离


        for region_type, distance_and_path_list in region_dict.items():
            total_distance = distance_and_path_list[0] #3537.917717097805
            test_distance = total_distance*test_ratio #884.4794292744513
            val_distance = total_distance*val_ratio #884.4794292744513

            path_list = distance_and_path_list[1] 
            tem_list = copy.deepcopy(path_list) # 会拷贝对象及其子对象,tmp_list 保存了原来path_list的内容，不会改变

            random.seed(2019)
            shuffle(tem_list) #打乱tem_list

            sum_distance = 0 

            # Test Set
            while sum_distance < test_distance*0.8:
                path = tem_list.pop() #弹出列表最后一个元素 列表长度-1
                sum_distance += path[3] #将distance加和
                dataset_div['test'][region_type][0] = dataset_div['test'][region_type][0] + path[3]
                dataset_div['test'][region_type][1].append(path)

            # Val Set
            while sum_distance < (test_distance + val_distance)*0.8:
                path = tem_list.pop()
                sum_distance += path[3]
                dataset_div['val'][region_type][0] = dataset_div['val'][region_type][0] + path[3]
                dataset_div['val'][region_type][1].append(path)

            # Train Set
            dataset_div['train'][region_type][0] = total_distance - sum_distance #剩余的都给train集
            dataset_div['train'][region_type][1] = tem_list

        color=['red','green','blue']
        ## Visualiaztion with respect to region
        fig, ax = plt.subplots(figsize=(30, 15)) #figsize设置子图的宽度和高度
        div_type = 'train' #我认为这句没用

        vis_txt_height = 800
        for div_type in ['train','val','test']:
            for region in ['skyscraper','suburban','shopping']:
                vis_path_list = dataset_div[div_type][region][1]
                for path in vis_path_list:
                    x = [path[1][0],path[2][0]] #记录起点和终点的x
                    y = [path[1][1],path[2][1]] #记录起点和终点的y

                    if region == 'skyscraper':
                        ax.plot(x, y, color='red', zorder=1, lw=3)
                    elif region == 'suburban':
                        ax.plot(x, y, color='blue', zorder=1, lw=3)
                    elif region == 'shopping':
                        ax.plot(x, y, color='green', zorder=1, lw=3)

                    ax.scatter(x, y,color='black', s=120, zorder=2)

                # Visualize distance text
                distance = dataset_div[div_type][region][0]
                if region == 'skyscraper':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='red')
                elif region == 'suburban':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='blue')
                elif region == 'shopping':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='green')
                vis_txt_height-=30  #text高度-30 文字依次向下排列

        plt.savefig('region.png', dpi=200)
        plt.close()

        ## Visualization with respect to train/val/test
        fig, ax = plt.subplots(figsize=(30, 15))
        div_type = 'train'
        vis_txt_height = 800
        for div_type in ['train','val','test']:
            for region in ['skyscraper','suburban','shopping']:
                vis_path_list = dataset_div[div_type][region][1]
                for path in vis_path_list:
                    x = [path[1][0],path[2][0]]
                    y = [path[1][1],path[2][1]]

                    if div_type == 'train':
                        ax.plot(x, y, color='red', zorder=1, lw=3)
                    elif div_type == 'val':
                        ax.plot(x, y, color='blue', zorder=1, lw=3)
                    elif div_type == 'test':
                        ax.plot(x, y, color='green', zorder=1, lw=3)

                    ax.scatter(x, y,color='black', s=120, zorder=2)

                # Visualize distance text
                distance = dataset_div[div_type][region][0]
                if div_type == 'train':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='red')
                elif div_type == 'val':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='blue')
                elif div_type == 'test':
                    ax.annotate(div_type+' - '+ region+': '+str(distance), (-800, vis_txt_height),fontsize=20,color='green')
                vis_txt_height-=30 #text高度-30 文字依次向下排列

                    #ax.annotate(txt, (x, y))
        plt.savefig('train_val_test.png', dpi=200)
        plt.close()

        return dataset_div #返回分组好的数据集
    def read_selection_label(self, label_type):

        if label_type == 'when2com':
            with open(os.path.join(self.root,'gt_when_to_communicate.txt')) as f:
                content = f.readlines()

            com_label = {}
            for x in content:
                key = x.split(' ')[2].strip().split('/')[-3] + '/' + x.split(' ')[2].strip().split('/')[-1]+'.png'
                com_label[key] = int(x.split(' ')[1])

        elif label_type == 'mimo': ##1111
            with open(os.path.join(self.root,'gt_mimo_communicate.txt')) as f:
                content = f.readlines() #将文件按行读取 type为list 
            com_label = {}
            for x in content: # x：'(0, 1, 0, 0, 0, 0) (4, 5, 2, 1, 3, 0) airsim_mimo_noise_v1/scene/async_rotate_fog_000_clear/0_0__16_74/view/frame000022\n'
                file_key = x.split(' ')[-1].strip().split('/')[-3] + '/' + x.split(' ')[-1].strip().split('/')[-1]+'.png' #'0_0__16_74/frame000022.png'
                noise_label = make_tuple(x.split(' (')[0]) #(0, 1, 0, 0, 0, 0)
                link_label = make_tuple(x.split(') ')[1] + ')') #(4, 5, 2, 1, 3, 0)
                com_label[file_key] = torch.tensor([noise_label,  link_label])

        else:
            raise ValueError('Unknown label file name '+ str(label_type))


        print('Loaded: selection label.')
        return com_label

    def convert_link_label(self, link_label):
        div_list = []
        for i in link_label:
            div_list.append(int(i/2))

        new_link_label = []
        for i, elem_i in enumerate(div_list):
            for j, elem_j in enumerate(div_list):
                if j != i and elem_i == elem_j:
                    new_link_label.append(j)
        new_link_label = tuple(new_link_label)
        return new_link_label
    def get_cam_pos(self, target_view):
        if target_view == "overhead":
            cam_pos = [ 'overhead', 'front',  'back', 'left', 'right']
        elif target_view == "front":
            cam_pos = [  'front',  'back', 'left', 'right','overhead']
        elif target_view == "back":
            cam_pos = [  'back',   'front',  'left', 'right','overhead']
        elif target_view == "left":
            cam_pos = [  'left',   'back',   'front','right','overhead']
        elif target_view == "target":
            cam_pos = [  'target',   'normal1', 'normal2','normal3','normal4']
        elif target_view == "6agent": #1111
            cam_pos = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5', 'agent6']
        elif target_view == "5agent":
            cam_pos = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']
        elif target_view == "DroneNP":
            cam_pos = ["DroneNN_main", "DroneNP_main", "DronePN_main", "DronePP_main", "DroneZZ_main"]
        elif target_view == "DroneNN_backNN":
            cam_pos = ["DroneNN_backNN", "DroneNP_backNP", "DronePN_backPN", "DroneNN_frontNN", "DroneNP_frontNP"]
        elif target_view == "5agentv7":
            cam_pos = ["agent1", "agent3", "agent5", "agent2", "agent4"]
        else:
            cam_pos = [ 'front',  'back', 'left', 'right', 'overhead']
        return cam_pos
    # <---- Functions for conversion of paths ----> 




    def __len__(self):
        """__len__"""
        return len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_list = [] 
        lbl_list = []

        for k, camera in enumerate(self.cam_pos):

            img_path, mask_path = self.imgs[self.split][camera]['scene'][index], self.imgs[self.split][camera]['segmentation_decoded'][index]
            img, mask = np.array(cv2.imread(img_path),dtype=np.uint8)[:,:,:3], np.array(cv2.imread(mask_path),dtype=np.uint8)[:,:,0]

            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lbl = mask 
            if self.augmentations is not None:
                img, lbl, aux = self.augmentations(img, lbl)

            if self.is_transform:
                img, lbl = self.transform(img, lbl)

            img_list.append(img)
            lbl_list.append(lbl)

        if self.commun_label != 'None':
            return img_list, lbl_list, self.com_label[self.split][index] #, self.debug_file_path[self.split][index]
        else:
            return img_list, lbl_list


    def transform(self, img, lbl):

        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for i,name in self.id2name.items():
            r[(temp==i)] = self.name2color[name][0][0]
            g[(temp==i)] = self.name2color[name][0][1]
            b[(temp==i)] = self.name2color[name][0][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


def save_tensor_imag(imgs):
    import numpy as np
    import cv2
    bs = imgs[0].shape[0]
    mean_rgb = np.array([103.939, 116.779, 123.68])

    for view in range(len(imgs)):
        for i in range(bs):
            image = imgs[view][i]

            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = image * 255 + mean_rgb
            cv2.imwrite('debug_tmp/img_b' + str(i) +'_v'+str(view)+ '.png', image)


