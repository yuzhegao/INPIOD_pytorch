# -*- coding:utf-8 –*- #
import torch
from torch.nn import functional as F
import numpy as np
import math
import cv2


def flow_warp(input, ori, size):
    out_h, out_w = size
    n, c, h, w = input.size()
    # n, c, h, w
    # n, 2, h, w
    pi = 3.141592654
#   ori = torch.div(ori, pi)
    flow = torch.empty(n, 2, out_h, out_w).type_as(ori).to(ori.device)
    flow[:, 0, :, :] = torch.cos(ori[:, :])
    flow[:, 1, :, :] = torch.sin(ori[:, :])
    flow = F.upsample(flow, size=(2 * h, 2 * w), mode="bilinear", align_corners=True)
    out_h, out_w = 2 * out_h, 2 * out_w
    norm = torch.tensor([[[out_w, out_h]]]).type_as(input).to(input.device)
    h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
    grid = grid.repeat(1, 1, 1).type_as(input).to(input.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm
    output = F.grid_sample(input, grid)
    output = F.interpolate(output, scale_factor=0.5, mode='bilinear')

    return output


def test_without_flow():
    """ 为了验证在不加flow的情况下，grid_sample可以原封不动地采样出原来的tensor """
    out_h = 4
    out_w = 3

    input_arr = torch.from_numpy(np.arange(out_h*out_w).reshape(1,1,out_h,out_w)).float()
    print(input_arr)

    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    # grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
    grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2) ## 需要注意一下，这里w和h的顺序是不能变的，否则会出错
    grid = grid.unsqueeze(0)
    # print(new_h.size(), new_w.size())
    # print(new_h)
    # print(new_w)
    print(grid)

    outp = F.grid_sample(input_arr, grid=grid, mode='bilinear', align_corners=True) ## 为了还原，这里的align_corners要设为True
    print(outp)

    print(input_arr.size(), outp.size())

def test_with_flow():
    """ 验证如何使用grid sample来进行残缺轮廓的补全(x对应w，y对应h) """
    out_h = 2
    out_w = 3
    input_array = np.array([0, 1, 0, 0, 0, 0]).reshape(1, 1, out_h, out_w)
    cv2.imwrite('input.png', np.squeeze(input_array) * 255)

    input_tensor = torch.from_numpy(input_array).float()

    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)

    flow_array_x = torch.from_numpy(np.array([1, 0, 0, 0, 0, 0]).reshape(out_h, out_w)).float()
    flow_array_y = torch.from_numpy(np.array([0, 0, 0, 0, -1, 0]).reshape(out_h, out_w)).float()
    ## 将绝对的像素值归一化到[-1,1]之间
    flow_array_x = flow_array_x / out_w
    flow_array_y = flow_array_y / out_h

    grid_flow = torch.cat((flow_array_x.unsqueeze(2), flow_array_y.unsqueeze(2)), dim=2) #[out_h, out_w,2]
    grid_flow = grid_flow.unsqueeze(0) #[1, out_h, out_w,2]
    print(grid_flow)
    grid = grid + grid_flow


    output_tensor = F.grid_sample(input_tensor, grid=grid, mode='bilinear', align_corners=True)
    # output_tensor = F.grid_sample(output_tensor, grid=grid, mode='bilinear', align_corners=True)
    # output_tensor = F.grid_sample(output_tensor, grid=grid, mode='bilinear', align_corners=True)
    output_array = np.squeeze(output_tensor.numpy())
    cv2.imwrite('output2.png', output_array * 255)


def test_torch_radius():
    """ 检查8个点的弧度取sin/cos，并检查对应关系是否正确 """
    def compute(x):
        cosine_value = torch.cos(torch.from_numpy(np.array([x])).float())
        sine_value   = torch.sin(torch.from_numpy(np.array([x])).float())
        print([cosine_value.numpy()[0], sine_value.numpy()[0]])

    radius_lst = [0, math.pi / 4, math.pi / 2, (3 * math.pi / 4), math.pi] ## 在x轴以下，弧度为正
    for x in radius_lst:
        compute(x)

    print('\n')
    radius_lst = [-math.pi / 4, -math.pi / 2, (-3 * math.pi / 4), math.pi] ## 在x轴以上，弧度为负
    for x in radius_lst:
        compute(x)


if __name__ == '__main__':
    torch.set_printoptions(profile="full", sci_mode=False)
    np.set_printoptions(suppress=True)

    test_with_flow()
    # test_torch_radius()

    print('\n')

