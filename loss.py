import torch
import torch.nn as nn
import torch.nn.functional as F


def objectness_module(name, args):
    assert name in ["CE", "weighted-CE", "focal", "reduced-focal"]
    if name == "CE":
        return WeightedBCELoss(alpha=args.alpha)
    if name == "weighted-CE":
        return WeightedBCELoss(pos_weight=1.0 - (1.0 / 77.0), alpha=args.alpha)
    if name == "focal":
        return BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
    if name == "reduced-focal":
        return BinaryFocalLoss(
            alpha=args.alpha, gamma=args.gamma, reduce_th=args.reduce_th
        )
    return None


class WeightedBCELoss(nn.Module):

    def __init__(self, pos_weight=0.5, alpha=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.eps = 1e-6

    def forward(self, input, target):
        assert input.size() == target.size()
        input = input[:, 0]
        target = target[:, 0]
        y_pred = input.contiguous().view(-1)
        y_true = target.contiguous().view(-1)
        weights = torch.where(
            y_true == 1.0,
            torch.ones_like(y_true) * self.pos_weight,
            torch.ones_like(y_true) * (1.0 - self.pos_weight),
        )
        y_pred = torch.clamp(y_pred, self.eps, 1.0)
        bce = F.binary_cross_entropy(y_pred, y_true, weight=weights, reduction="sum")
        return self.alpha * 2.0 * bce / torch.sum(target)


class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=1.0, reduce_th=0.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce_th = reduce_th
        self.eps = 1e-6

    def forward(self, input, target):
        assert input.size() == target.size()
        input = input[:, 0]
        target = target[:, 0]
        y_pred = input.contiguous().view(-1)
        y_true = target.contiguous().view(-1)
        y_pred = torch.clamp(y_pred, self.eps, 1.0)
        log_pt = -F.binary_cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(log_pt)
        th_pt = torch.where(
            pt < self.reduce_th,
            torch.ones_like(pt),
            (((1 - pt) / (1 - self.reduce_th)) ** self.gamma),
        )
        loss = -self.alpha * th_pt * log_pt
        return torch.sum(loss) / torch.sum(target)


class LocalizationLoss(nn.Module):

    def __init__(self, weight=1.0):
        super(LocalizationLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        assert input.size() == target.size()
        mask = torch.where(
            target[:, 0] == 1.0, target[:, 0], torch.zeros_like(target[:, 0])
        )
        input_y = input[:, 1] * mask
        input_x = input[:, 2] * mask
        input_h = input[:, 3] * mask
        input_w = input[:, 4] * mask
        input_offset = torch.cat((input_y, input_x), dim=1)
        input_box = torch.cat((input_h, input_w), dim=1)
        target_offset = target[:, [1, 2]]
        target_box = target[:, [3, 4]]
        y_pred_offset = input_offset.view(input_offset.size()[0], -1)
        y_true_offset = target_offset.view(target_offset.size()[0], -1)
        y_pred_box = input_box.view(input_box.size()[0], -1)
        y_true_box = target_box.view(target_box.size()[0], -1)
        loss_offset = F.mse_loss(y_pred_offset, y_true_offset, reduction="sum")
        loss_offset = loss_offset / torch.sum(mask)
        y_pred_box = torch.where(
            y_true_box == 0.0, torch.zeros_like(y_pred_box), y_pred_box
        )
        loss_box = F.mse_loss(y_pred_box, y_true_box, reduction="sum")
        loss_box = loss_box / torch.sum(mask)
        return (loss_offset + loss_box) * self.weight
