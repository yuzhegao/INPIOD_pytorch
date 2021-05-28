# -*- coding:utf-8 –*- #
import argparse
import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm, trange
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', required=True, help='Prediction directory')
opt = parser.parse_args()

lst_file = '/home/gyz/data/INPIOD/val_INPIOD.txt'
gt_root = '/home/gyz/data/INPIOD/Annotation_mat/'
pred_root = '/home/gyz/document3/DCNet_pytorch2/experiments/output/' + opt.pred_dir + '/result_insboundary/'
# pred_root = '/home/gyz/document3/DCNet_pytorch2/experiments/output/2021-03-18_14-22-28/result_insboundary/'



def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice

def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return max(_dice1, _dice2)


def calc_miou(ins_seg_gt, ins_seg_pred):
    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_ious = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        ious = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            iou = dice / (2 - dice)
            ious.append(iou)
        best_iou = np.max(ious)
        best_ious.append(best_iou)

    best_iou = np.mean(best_ious)

    return best_iou





def single_test_with_edgeGT(id):
    pred_file = pred_root + id + '-ins_mask.png'
    gt_file = gt_root + id + '.mat'

    m = loadmat(gt_file)
    ins_anno = m['ins_anno']  # (H, W, num_ins)
    ins_anno = ins_anno.astype(np.float32)

    # print(ins_anno.shape)
    gt_ins_map = np.zeros([ins_anno.shape[0], ins_anno.shape[1]]).astype(np.uint8) # [H,W]
    for i in range(ins_anno.shape[2]):
        ins_map = ins_anno[:, :, i]
        gt_ins_map[ins_map == 1] = int(i + 1)

    check_gt = False
    if check_gt:
        ins_seg_pred_pil = Image.fromarray(gt_ins_map.astype(np.uint8))
        ins_seg_pred_pil.save(id + '-ins_mask.png')
        ins_pred = np.array(Image.open('2008_000009-ins_mask.png'))
        # print(ins_pred.shape)
        cv2.imwrite('1.png', (ins_pred==2)*255)

    pred_ins_map = np.array(Image.open(pred_file))

    sbd = calc_sbd(gt_ins_map, pred_ins_map)
    iou = calc_miou(gt_ins_map, pred_ins_map)

    return sbd, iou


def Eval():
    with open(lst_file, 'r') as f:
        lines = f.readlines()
        id_list = [line.rstrip() for line in lines]

    sbd_list, iou_list = [], []
    for id in tqdm(id_list):
        sbd, iou = single_test_with_edgeGT(id)
        sbd_list.append(sbd)
        iou_list.append(iou)

    print('Instance Evaluate Result: SBD={:.3f} mIOU={:.3f}'.format(np.mean(sbd_list), np.mean(iou_list)))



if __name__ == '__main__':
    Eval()

