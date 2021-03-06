# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
import torch
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import h5py
from scipy.io import loadmat
import torchvision.transforms as F

PI = 3.1416


class ImgToOccDataset(data.Dataset):
    """generic dataset loader for occlusion edge/ori/order estimation from image"""

    def __init__(self, csv_file, config, isTest=False, input_transf=None, target_transf=None, co_transf=None):
        self.samples = pd.read_csv(csv_file)
        self.input_transform = input_transf
        self.target_transform = target_transf
        self.co_transform = co_transf
        self.config = config
        self.isTest = isTest

    def __getitem__(self, idx):
        # load sample data
        img_path = os.path.join(self.config.root_path, self.samples.iloc[idx, 0])
        occ_edge_lbl_path = os.path.join(self.config.root_path, self.samples.iloc[idx, 1])
        occ_ori_lbl_path = os.path.join(self.config.root_path, self.samples.iloc[idx, 2])
        occ_order_lbl_path = os.path.join(self.config.root_path, self.samples.iloc[idx, 3])

        img = Image.open(img_path, 'r')
        img_org_sz = [img.size[1], img.size[0]]  # H,W
        if self.isTest and occ_order_lbl_path.split('/')[-1] == ' ':  # test w/o GT
            occ_edge_lbl = np.ones((img_org_sz[0], img_org_sz[1]))
            occ_ori_lbl = np.ones((img_org_sz[0], img_org_sz[1]))
            occ_order_lbl = np.ones((img_org_sz[0], img_org_sz[1], 9))
        else:
            occ_edge_lbl = cv2.imread(occ_edge_lbl_path, cv2.IMREAD_UNCHANGED) / 255  # H,W; {0, 1}
            occ_ori_lbl = cv2.imread(occ_ori_lbl_path, cv2.IMREAD_UNCHANGED) / 255  # H,W; {0, 1}
            occ_order_lbl = np.load(occ_order_lbl_path)['order']  # H,W,9; {-1,0,1}

        # make net load size as multiplier of minSize
        if self.isTest:
            H_load, W_load = get_net_loadsz(img_org_sz, self.config.network.scale_down)
            if self.config.TEST.img_padding:  # reflect pad for paired img size
                pad_H = int((H_load - img_org_sz[0]) / 2)
                pad_W = int((W_load - img_org_sz[1]) / 2)
                reflect_pad = F.Pad((pad_W, pad_H), padding_mode='reflect')
                img = reflect_pad(img)
            else:  # resize img to fit net input size
                img = img.resize((W_load, H_load), Image.LANCZOS)

        if self.config.network.task_type == 'occ_order':
            # add edge-wise, 3-bin classification for each pix-pair edge
            occ_order_lbl_E = (occ_order_lbl[:, :, 5] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
            occ_order_lbl_S = (occ_order_lbl[:, :, 7] + 1).astype(np.int8)
            occ_lbl_list = [occ_order_lbl_E, occ_order_lbl_S]
            if self.config.dataset.connectivity == 8:
                occ_order_lbl_SE = (occ_order_lbl[:, :, 8] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
                occ_order_lbl_NE = (occ_order_lbl[:, :, 3] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
                occ_lbl_list += [occ_order_lbl_SE, occ_order_lbl_NE]

            if self.config.TRAIN.mask_is_edge:  # define pix occ edge as pix where occ order exists
                occ_order_lbl_abs = np.abs(occ_order_lbl[:, :, 1:])  # H,W,8
                order_exist_lbl = (np.sum(occ_order_lbl_abs, axis=2) > 0).astype(np.float)
                occ_lbl_list += [order_exist_lbl]
            else:
                occ_lbl_list += [occ_edge_lbl]
        elif self.config.network.task_type == 'occ_ori':  # add occ ori label
            occ_ori_lbl = (occ_ori_lbl * 2 - 1) * PI  # [0,1] => [-PI,PI]
            occ_lbl_list = [occ_ori_lbl, occ_edge_lbl]

        # transforms and data augmentation
        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: occ_lbl_list = self.target_transform(occ_lbl_list)
        if self.co_transform is not None:
            occ_lbl_list = [Image.fromarray(label) for label in occ_lbl_list]
            input_pair = {'image': img, 'label': occ_lbl_list}
            input_pair = self.co_transform(input_pair)
            img = input_pair['image']
            occ_lbl_list = input_pair['label']
        sample = ((img), tuple(occ_lbl_list), img_path)

        return sample

    def __len__(self):
        return self.samples.__len__()


class INPIOD_Dataset(data.Dataset):
    """generic dataset loader for occlusion edge/ori/order estimation from image"""

    def __init__(self, config, isTest=False, input_transf=None, target_transf=None, co_transf=None,
                 edge_only=False, max_n_objects=16):
        self.input_transform = input_transf
        self.target_transform = target_transf
        self.co_transform = co_transf
        self.config = config
        self.isTest = isTest

        self.edge_only = edge_only
        self.max_n_objects = max_n_objects

        self.img_root = os.path.join(config.dataset.train_image_set, 'JPEGImages')
        self.label_root = os.path.join(config.dataset.train_image_set, 'Annotation_mat')
        if not self.isTest:
            list_file = os.path.join(config.dataset.train_image_set, 'train_INPIOD.txt')
        else:
            list_file = os.path.join(config.dataset.train_image_set, 'val_INPIOD.txt')

        with open(list_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.rstrip()
        self.img_list = sorted(lines)

    def __getitem__(self, idx):
        # load sample data
        img_path = os.path.join(self.img_root, self.img_list[idx] + '.jpg')

        img = Image.open(img_path, 'r')

        # transforms and data augmentation
        label_filename = os.path.join(self.label_root, self.img_list[idx] + '.mat')
        m = loadmat(label_filename)
        sem_anno = m['sem_anno']  # (H, W)
        ins_anno = m['ins_anno']  # (H, W, num_ins)
        sem_anno = sem_anno.astype(np.float32)
        ins_anno = ins_anno.astype(np.float32)

        n_objects = ins_anno.shape[2]
        ins_anno_l = [ins_anno[:, :, i] for i in range(ins_anno.shape[2])]  # list: num_ins * (H, W)

        occ_lbl_list = [sem_anno] + ins_anno_l  # list: [0] sem; [1:] ins

        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: occ_lbl_list = self.target_transform(occ_lbl_list)
        if self.co_transform is not None:
            occ_lbl_list = [Image.fromarray(label) for label in occ_lbl_list]
            input_pair = {'image': img, 'label': occ_lbl_list}
            input_pair = self.co_transform(input_pair)
            img = input_pair['image']
            occ_lbl_list = input_pair['label']

        sem_anno = occ_lbl_list[0]
        ins_anno_l = occ_lbl_list[1:]
        for i in range(self.max_n_objects - n_objects):
            zero = torch.zeros(sem_anno.shape)
            ins_anno_l.append(zero)
        ins_anno = torch.tensor([item.numpy() for item in ins_anno_l])

        sample = (img, sem_anno, ins_anno, n_objects, img_path)

        return sample

    def __len__(self):
        return len(self.img_list)


class PIOD_Dataset(data.Dataset):
    """generic dataset loader for occlusion edge/ori/order estimation from image"""

    def __init__(self, config, isTest=False, input_transf=None, target_transf=None, co_transf=None, edge_only=False):
        self.input_transform = input_transf
        self.target_transform = target_transf
        self.co_transform = co_transf
        self.config = config
        self.isTest = isTest

        self.edge_only = edge_only

        self.img_root = os.path.join(config.dataset.train_image_set, 'Augmentation', 'Aug_JPEGImages')
        self.label_root = os.path.join(config.dataset.train_image_set, 'Augmentation', 'Aug_HDF5EdgeOriLabel')
        if not self.isTest:
            list_file = os.path.join(config.dataset.train_image_set, 'Augmentation', 'train_pair_320x320.lst')
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    lines[i] = (os.path.split(line.rstrip().split()[0])[1])[:-4]
        else:
            list_file = os.path.join(config.dataset.train_image_set, 'val_doc_2010.txt')
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    lines[i] = line.rstrip()
        self.img_list = sorted(lines)

    def __getitem__(self, idx):
        # load sample data
        img_path = os.path.join(self.img_root, self.img_list[idx] + '.jpg')
        # occ_edge_lbl_path  = os.path.join(self.config.root_path, self.samples.iloc[idx, 1])
        # occ_ori_lbl_path   = os.path.join(self.config.root_path, self.samples.iloc[idx, 2])
        # occ_order_lbl_path = os.path.join(self.config.root_path, self.samples.iloc[idx, 3])

        img = Image.open(img_path, 'r')
        img_org_sz = [img.size[1], img.size[0]]  # H,W
        # if self.isTest and occ_order_lbl_path.split('/')[-1] == ' ':  # test w/o GT
        #     occ_edge_lbl  = np.ones((img_org_sz[0], img_org_sz[1]))
        #     occ_ori_lbl   = np.ones((img_org_sz[0], img_org_sz[1]))
        #     occ_order_lbl = np.ones((img_org_sz[0], img_org_sz[1], 9))
        # else:
        #     occ_edge_lbl  = cv2.imread(occ_edge_lbl_path, cv2.IMREAD_UNCHANGED) / 255  # H,W; {0, 1}
        #     occ_ori_lbl   = cv2.imread(occ_ori_lbl_path, cv2.IMREAD_UNCHANGED) / 255  # H,W; {0, 1}
        #     occ_order_lbl = np.load(occ_order_lbl_path)['order']  # H,W,9; {-1,0,1}

        # make net load size as multiplier of minSize
        # if self.isTest:
        #     H_load, W_load = get_net_loadsz(img_org_sz, self.config.network.scale_down)
        #     if self.config.TEST.img_padding:  # reflect pad for paired img size
        #         pad_H = int((H_load - img_org_sz[0]) / 2)
        #         pad_W = int((W_load - img_org_sz[1]) / 2)
        #         reflect_pad = F.Pad((pad_W, pad_H), padding_mode='reflect')
        #         img = reflect_pad(img)
        #     else:  # resize img to fit net input size
        #         img = img.resize((W_load, H_load), Image.LANCZOS)

        # if self.config.network.task_type == 'occ_order':
        #     # add edge-wise, 3-bin classification for each pix-pair edge
        #     occ_order_lbl_E = (occ_order_lbl[:, :, 5] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
        #     occ_order_lbl_S = (occ_order_lbl[:, :, 7] + 1).astype(np.int8)
        #     occ_lbl_list = [occ_order_lbl_E, occ_order_lbl_S]
        #     if self.config.dataset.connectivity == 8:
        #         occ_order_lbl_SE = (occ_order_lbl[:, :, 8] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
        #         occ_order_lbl_NE = (occ_order_lbl[:, :, 3] + 1).astype(np.int8)  # [-1, 0, 1] => [0, 1, 2]
        #         occ_lbl_list += [occ_order_lbl_SE, occ_order_lbl_NE]
        #
        #     if self.config.TRAIN.mask_is_edge:  # define pix occ edge as pix where occ order exists
        #         occ_order_lbl_abs = np.abs(occ_order_lbl[:, :, 1:])  # H,W,8
        #         order_exist_lbl = (np.sum(occ_order_lbl_abs, axis=2) > 0).astype(np.float)
        #         occ_lbl_list += [order_exist_lbl]
        #     else:
        #         occ_lbl_list += [occ_edge_lbl]
        # elif self.config.network.task_type == 'occ_ori':  # add occ ori label
        #     occ_ori_lbl  = (occ_ori_lbl * 2 - 1) * PI  # [0,1] => [-PI,PI]
        #     occ_lbl_list = [occ_ori_lbl, occ_edge_lbl]

        # transforms and data augmentation
        label_filename = os.path.join(self.label_root, self.img_list[idx] + '.h5')
        h5 = h5py.File(label_filename, 'r')
        label_occ = np.squeeze(h5['label'][...])
        label_occ = np.transpose(label_occ, axes=(1, 2, 0))  ## (H,W,2)   0-edgemap 1-orientmap
        occ_lbl_list = [label_occ[:, :, 0], label_occ[:, :, 1]]

        if self.input_transform is not None: img = self.input_transform(img)
        if self.target_transform is not None: occ_lbl_list = self.target_transform(occ_lbl_list)
        if self.co_transform is not None:
            occ_lbl_list = [Image.fromarray(label) for label in occ_lbl_list]
            input_pair = {'image': img, 'label': occ_lbl_list}
            input_pair = self.co_transform(input_pair)
            img = input_pair['image']
            occ_lbl_list = input_pair['label']
        sample = ((img), tuple(occ_lbl_list), img_path)

        return sample

    def __len__(self):
        return len(self.img_list)


def get_net_loadsz(org_img_sz, minSize):
    """
    for net with input constrain
    :param org_img_sz: [H, W]
    :param minSize: int
    :return: net load size
    """
    H_load = (org_img_sz[0] // minSize) * minSize
    W_load = (org_img_sz[1] // minSize) * minSize
    if H_load < org_img_sz[0]: H_load += minSize
    if W_load < org_img_sz[1]: W_load += minSize

    return H_load, W_load
