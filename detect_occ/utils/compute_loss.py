# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# define loss terms for occlusion edge/order detection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

PI = 3.1416


def get_criterion(config, args):
    """get relevant loss for train"""
    if config.TRAIN.loss == 'CrossEntropyLoss':
        cls_weights = torch.tensor(config.TRAIN.class_weights, dtype=torch.float32)
        criterion = CCELoss(config, args.gpus[0], cls_weights=cls_weights,
                            spatial_weights=config.TRAIN.spatial_weighting, size_average=True)
    elif config.TRAIN.loss == 'AL_and_L1':
        criterion = AL_and_L1(config)
    return criterion


def cal_loss(criterion, net_out, targets, config):
    """
    :param net_out: N,C,H,W
    :param targets: occ order target [occ_order_1, ..., occ_order_C, occ_edge]
                    occ ori target   [occ_ori, occ_edge]
    """
    if config.network.task_type == 'occ_order':
        occ_order_loss_E = criterion(net_out[:, 0:3, :, :], targets[0], targets[-1])
        occ_order_loss_S = criterion(net_out[:, 3:6, :, :], targets[1], targets[-1])
        total_loss = (occ_order_loss_E + occ_order_loss_S)

        if config.dataset.connectivity == 4:
            return total_loss, [occ_order_loss_E, occ_order_loss_S]
        elif config.dataset.connectivity == 8:
            occ_order_loss_SE = criterion(net_out[:, 6:9, :, :], targets[2], targets[-1])
            occ_order_loss_NE = criterion(net_out[:, 9:12, :, :], targets[3], targets[-1])
            total_loss += (occ_order_loss_SE + occ_order_loss_NE)
            return total_loss, [occ_order_loss_E, occ_order_loss_S, occ_order_loss_SE, occ_order_loss_NE]

    elif config.network.task_type == 'occ_ori':
        occ_edge_loss, occ_ori_loss = criterion(net_out, targets)
        occ_edge_loss = config.TRAIN.loss_gamma[0] * occ_edge_loss
        occ_ori_loss = config.TRAIN.loss_gamma[1] * occ_ori_loss
        total_loss = (occ_edge_loss + occ_ori_loss)

        return total_loss, [occ_edge_loss, occ_ori_loss]


###########################################################
## for occlusion edge
class OFloss(nn.Module):
    def __init__(self):
        super(OFloss, self).__init__()
        self.beta = 4.0
        self.gamma = 0.5
        self.sigma = 0.5
        self.lossinfo = []

    def forward(self, output, label, config):
        # label_edge, label_ori = label[0], label[1]
        # output_edge, output_ori = output[0], output[1]
        # # edge_loss = self.attention_loss_fromdoob(output_edge, label_edge)
        # edge_loss = self.attention_loss(output_edge, label_edge.float())
        # ori_loss = self.smoothL1_loss(output_ori, label_ori, label_edge)
        # self.lossinfo = [edge_loss.item(), ori_loss.item()]
        # return 1.0 * edge_loss + 0.5 * ori_loss, [edge_loss.item(), ori_loss.item()]
        edge_loss = self.attention_loss(output, label.float())
        return edge_loss

    def attention_loss(self, output, target):
        N = output.shape[0]
        target = target.unsqueeze(1)
        alpha = 1.0 - torch.mean(target)
        diff = torch.log(torch.exp(-torch.abs(output)) + 1) + torch.relu(-output)
        eps = 1e-5
        p = torch.clamp(torch.sigmoid(output), eps, 1 - eps)
        cost = alpha * target * diff * torch.pow(self.beta, torch.pow(1 - p, self.gamma)) + \
               (1 - alpha) * (1 - target) * (output + diff) * torch.pow(self.beta, torch.pow(p, self.gamma))
        return cost.sum() / N

    def attention_loss_fromdoob(self, output, target):
        beta = Variable(torch.tensor(float(self.beta))).cuda()
        gamma = Variable(torch.tensor(float(self.gamma))).cuda()
        al = Attention_loss()
        return al.apply(output, target, beta, gamma)

    def smoothL1_loss(self, output, target, weight):
        N = output.shape[0]
        x = self._theta_penalize(target, output, weight)
        return torch.sum(torch.where(torch.abs(x) < 1 / self.sigma ** 2, 0.5 * (x * self.sigma) ** 2,
                                     torch.abs(x) - (0.5 / self.sigma ** 2))) / N

    def _theta_penalize(self, gt, output, weight):
        pi = 3.141592654
        weight = weight.unsqueeze(1).bool()
        gt = gt.unsqueeze(1)[weight]
        output = output[weight]
        indexs = (((gt > 0) & (output > pi)) | ((gt < 0) & (output < -pi)))
        indexs_not = torch.logical_not(indexs)
        return torch.cat([output[indexs_not] - gt[indexs_not], output[indexs] + gt[indexs]])


###########################################################
## for occlusion edge cluster
class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True, size_average=True):
        super(DiscriminativeLoss, self).__init__(size_average)

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        #         _assert_no_grad(target)
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        max_n_clusters = target.size(1)

        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, :n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / (target_sample.sum(2) + 1e-15)  #  avoid 0

            # padding
            n_pad_clusters = max_n_clusters - n_clusters[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters)
                pad_sample = Variable(pad_sample)
                if self.usegpu:
                    pad_sample = pad_sample.cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(bs):
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / (target_sample.sum(1) + 1e-15)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin)
            if self.usegpu:
                margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term

class nonboundary_regular(_Loss):
    def __init__(self, mode='l1norm', size_average=True):
        super(nonboundary_regular, self).__init__(size_average)

        self.mode = mode
        if mode not in ['l1', 'l2', 'l1norm']:
            assert 'regular loss is not in mode'

    def l1_loss(self, out_ins, sem_anno):
        N, C = out_ins.shape[0], out_ins.shape[1]

        diff = (1.0 - sem_anno) * torch.mean(torch.abs(out_ins), dim=1)

        return torch.sum(diff) / N

    def l2_loss(self, out_ins, sem_anno):
        N, C = out_ins.shape[0], out_ins.shape[1]

        diff = (1.0 - sem_anno) * torch.mean(out_ins ** 2, dim=1)

        return torch.sum(diff) / N

    def l1norm_loss(self, out_ins, sem_anno):
        N, C = out_ins.shape[0], out_ins.shape[1]

        norm = torch.norm(out_ins, 2, 1)
        diff = (1.0 - sem_anno) * norm

        return torch.sum(diff) / N

    def forward(self, out_ins, sem_anno):
        """
         out_ins:  [N,32,H,W]
         sem_anno: [N,H,W]
        """
        cost = 0
        if self.mode == 'l1':
            cost = self.l1_loss(out_ins, sem_anno)
        if self.mode == 'l2':
            cost = self.l2_loss(out_ins, sem_anno)
        if self.mode == 'l1norm':
            cost = self.l1norm_loss(out_ins, sem_anno)

        return cost






class Attention_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, target, beta, gamma):
        N = output.shape[0]
        target = target.unsqueeze(1).float()
        pos_count = torch.sum(target)
        neg_count = torch.sum(1.0 - target)
        alpha = Variable(neg_count / (neg_count + pos_count))

        sigma = torch.log(torch.exp(- torch.abs(output)) + 1) + torch.relu(- output)
        oriloss = - (target * sigma + (1 - target) * (output + sigma))
        p = torch.sigmoid(output)
        scale = alpha * target * torch.pow(beta, torch.pow(1 - p, gamma)) + \
                (1 - alpha) * (1 - target) * torch.pow(beta, torch.pow(p, gamma))
        cost = - scale * oriloss
        ctx.save_for_backward(oriloss, target, alpha, p, scale, beta, gamma)
        return cost.sum() / N

    @staticmethod
    def backward(ctx, grad_output):
        oriloss, target, alpha, p, scale, beta, gamma = ctx.saved_tensors
        N = p.shape[0]
        temp = - target * p + (1 - target) * ((1 - p) * oriloss * torch.log(beta) * gamma * torch.pow(p, gamma - 1) + 1)
        grad = scale * temp * (target * (1 - p) + (1 - target) * p) / N
        return grad_output * grad, None, None, None


class Attention_loss_doob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, target, beta, gamma):
        N = output.shape[0]
        target = target.unsqueeze(1).float()
        pos_count = torch.sum(target)
        neg_count = torch.sum(1.0 - target)
        alpha = Variable(neg_count / (neg_count + pos_count))

        p = torch.sigmoid(output)
        # act_neg = torch.sum((p > mu_neg).float() * (1.0 - target))
        # act_pos = torch.sum((p < mu_pos).float() * target)
        # alpha = Variable(torch.clamp(mu_pop * act_pos / (act_neg + act_pos), act_neg / (act_neg + act_pos), 0.9999))
        eps = 1e-8
        p = p.clamp(eps, 1 - eps)
        pred_pos = (output >= 0).float()
        scale = target * alpha * beta ** ((1.0 - p) ** gamma) + (1.0 - target) * (1.0 - alpha) * beta ** (p ** gamma)
        oriloss = -(output * (target - pred_pos) - torch.log(1 + torch.exp(output - 2 * output * pred_pos)))

        cost = scale * oriloss
        ctx.save_for_backward(oriloss, target, alpha, p, scale, beta, gamma)
        return cost.sum() / N

    @staticmethod
    def backward(ctx, grad_output):
        oriloss, target, alpha, p, scale, beta, gamma = ctx.saved_tensors
        N = p.shape[0]
        temp = - target * p + (1.0 - target) * ((1 - p) * oriloss * torch.log(beta) * gamma * p ** (gamma - 1) + 1)
        grad = scale * temp * (target * (1 - p) + (1 - target) * p) / N
        return grad_output * grad, None, None, None, None, None, None


def get_criterion_occ():
    """get relevant loss for train"""
    return OFloss()

def get_criterion_ins():
    return DiscriminativeLoss()

def get_criterion_reg():
    return nonboundary_regular()

class CCELoss(nn.Module):
    """class-balanced cross entropy loss for classification with given class weights"""

    def __init__(self, config, gpu_id, cls_weights='None', spatial_weights='None', size_average=True):
        super(CCELoss, self).__init__()
        self.config = config
        self.spatial_weights = spatial_weights
        self.size_average = size_average
        self.loss_CE = nn.CrossEntropyLoss(weight=cls_weights, reduction='none').cuda(gpu_id)

    def forward(self, net_out, target, mask):
        """
        :param net_out: N,C,H,W
        :param target: N,H,W; [0,1,2]
        :param mask: N,H,W; [0,1]
        :return:
        """
        loss = self.loss_CE(net_out, target)  # N,C,H,W => N,H,W
        loss = loss.view(-1)  # N,H,W => N*H*W,

        if self.spatial_weights != 'None':
            if self.config.TRAIN.mask_is_edge:  # where on occlusion edge pixels
                weight_mask = mask
            else:  # where curr occlusion order exists
                weight_mask = (target.clone().detach() != 1)

            weight_mask = weight_mask.view(-1).float()  # N,H,W => N*H*W,
            weight_mask[weight_mask == 1.] = self.spatial_weights[1]
            weight_mask[weight_mask == 0.] = self.spatial_weights[0]

            loss = weight_mask * loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class AL_and_L1(nn.Module):
    """Attention loss + SmoothL1 for occlusion edge/ori estimation"""

    def __init__(self, config):
        super(AL_and_L1, self).__init__()
        self.AttentionLoss = AttentionLoss(config.TRAIN.attentionloss_gamma_beta, avg_method='batch')
        self.SmoothL1Loss = SmoothL1Loss(config.TRAIN.smoothL1_sigma, avg_method='batch')

    def forward(self, net_out, targets):
        occ_edge_pred = net_out[:, 0, :, :].unsqueeze(dim=1)  # N,1,H,W
        occ_ori_pred = net_out[:, 1, :, :].unsqueeze(dim=1)
        occ_edge_gt = targets[1].unsqueeze(dim=1)
        occ_ori_gt = targets[0].unsqueeze(dim=1)

        occ_edge_loss = self.AttentionLoss(occ_edge_pred, occ_edge_gt)
        occ_ori_loss = self.SmoothL1Loss(occ_ori_pred, occ_ori_gt, occ_edge_gt)

        return occ_edge_loss, occ_ori_loss


class FocalLoss(nn.Module):
    """
    focal loss for classification task
    derived from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0., alpha='None', size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # default:2 ; for RetinaNet: [0.5, 5]
        self.alpha = alpha  # hyper-param(list) or inverse class frequency
        if isinstance(alpha, (float, int)):  # binary case
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, net_out, target, weighting_map):
        cls_ind = torch.arange(0, net_out.size(1), dtype=torch.long).tolist()
        if net_out.dim() > 2:
            net_out = net_out.view(net_out.size(0), net_out.size(1), -1)  # N,C,H,W => N,C,H*W
            net_out = net_out.transpose(1, 2)  # N,C,H*W => N,H*W,C
            net_out = net_out.contiguous().view(-1, net_out.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N,H,W => N*H*W,1

        log_pt = F.log_softmax(net_out, dim=1)  # log(softmax(net_out))
        log_pt = log_pt.gather(1, target)  # N*H*W,C => N*H*W,1 gather along axis 1
        log_pt = log_pt.view(-1)  # N*H*W,
        pt = log_pt.data.exp()  # softmax(net_out)

        if self.alpha == 'None':
            # use mini-batch inverse class frequency as alpha
            cls_num = torch.tensor([target.cpu().eq(cls_idx).sum() for cls_idx in cls_ind])
            # alpha = target.cpu().numel() / cls_num.float()  # C,
            alpha = 1 + torch.log(target.cpu().numel() / cls_num.float())  # C,
        else:
            alpha = self.alpha
        if alpha.type() != net_out.data.type():
            alpha = alpha.type_as(net_out.data)

        at = alpha.gather(0, target.data.view(-1))  # N*H*W,
        log_pt = log_pt * at  # alpha * log(softmax(net_out))
        loss = -1 * ((1 - pt) ** self.gamma) * log_pt  # N*H*W,
        if self.size_average:
            return loss.mean()  # average over each loss elem
        else:
            return loss.sum()


class AttentionLoss(nn.Module):
    """
    binary attention loss introduced in DOOBNet https://arxiv.org/pdf/1806.03772.pdf
    extension of focal loss by adding modulating param beta
    """

    def __init__(self, gamma_beta=(0.5, 4), alpha=None, size_average=True, avg_method='batch'):
        """
        :param gamma_beta:
        :param alpha: None or a float
        :param size_average:
        """
        super(AttentionLoss, self).__init__()
        self.gamma = gamma_beta[0]
        self.beta = gamma_beta[1]
        self.alpha = alpha
        self.size_average = size_average
        self.avg_method = avg_method

    def forward(self, net_out, target):
        """
        :param net_out:# net_out: (N, 1, H, W) ; activation passed by sigmoid [0~1]
        :param target: (N, 1, H, W)
        :return:
        """
        N, C, H, W = target.shape
        assert net_out.size(1) == 1

        # create mask to identify pixels at boundary
        edge = (target == 1).float()
        non_edge = (target != 1).float()

        if self.alpha is None:
            alpha = non_edge.sum() / (non_edge.sum() + edge.sum())
        else:
            alpha = self.alpha

        net_out = torch.clamp(net_out, 0.00000001, 0.99999999)  # according to caffe code

        scale_edge = alpha * torch.pow(self.beta, torch.pow((1 - net_out), self.gamma))
        scale_nonedge = (1 - alpha) * torch.pow(self.beta, torch.pow(net_out, self.gamma))

        log_p = net_out.log()
        log_m_p = (1 - net_out).log()

        loss = - edge * scale_edge * log_p - non_edge * scale_nonedge * log_m_p

        if self.size_average:
            if self.avg_method == 'batch':
                loss = loss.view(N, -1).sum(-1)
            return loss.mean()
        else:
            return loss.sum()  # too big loss may result in too big grad


class SmoothL1Loss(nn.Module):
    def __init__(self, sigma, avg_method):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.avg_method = avg_method

    def forward(self, pred, gt, mask, size_average=True):
        """
        compute smoothL1 loss for orientation regression on given mask=1 pixels as in DOOBNet
        :param pred: N,1,H,W; float
        :param gt: N,1,H,W
        :param mask: N,1,H,W; [0, 1.]; int
        :param type:
        :return:
        """
        N, C, H, W = gt.shape
        valid_elem = (mask == 1.).sum()
        gt = gt.float()
        mask = mask.float()

        mask_sum_pos = (pred > PI) * (gt > 0.)
        mask_sum_neg = (pred < -PI) * (gt <= 0.)

        sum = pred + gt
        diff = pred - gt
        x = diff
        x[mask_sum_pos] = sum[mask_sum_pos]
        x[mask_sum_neg] = sum[mask_sum_neg]

        x = x * mask
        x_abs = torch.abs(x)

        mask_in = (x_abs < (1 / self.sigma ** 2)).float()  # caffe code thresh: 1 / sigma**2; but thresh 1 works better
        mask_out = (x_abs >= (1 / self.sigma ** 2)).float()
        loss_in = 0.5 * (self.sigma ** 2) * (x ** 2)
        loss_out = x_abs - 0.5 / (self.sigma ** 2)

        loss = loss_in * mask_in + loss_out * mask_out

        if size_average:
            if self.avg_method == 'batch':
                loss = torch.sum(loss.view(N, -1), dim=1)
                return loss.mean()
            elif self.avg_method == 'mean':
                return loss.sum() / valid_elem
        else:
            return loss.sum()
