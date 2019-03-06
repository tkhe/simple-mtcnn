import torch
import torch.nn as nn
import torch.nn.functional as F

from mtcnn.config import cfg

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = torch.Tensor([1 - alpha, alpha])
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if target.is_cuda:
            self.alpha = self.alpha.cuda()
        at = self.alpha.gather(0, target.view(-1))
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()


class Loss(object):
    def __init__(self):
        if cfg.MODEL.FOCAL_LOSS_ON:
            self.loss_cls = FocalLoss(cfg.FOCAL_LOSS.ALPHA, cfg.FOCAL_LOSS.GAMMA)
        else:
            self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reg = nn.SmoothL1Loss()

    def __call__(self, cls, reg, label, target):
        cls_loss = self.cls_loss(cls, label)
        reg_loss = self.reg_loss(reg, target, label)
        return cls_loss, reg_loss

    def cls_loss(self, input, target):
        input = torch.squeeze(input)
        target = torch.squeeze(target)

        input = input[target >= 0]
        target = target[target >= 0]

        return self.loss_cls(input, target)

    def reg_loss(self, input, target, gt_label):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        gt_label = torch.squeeze(gt_label)

        input = input[gt_label != 0]
        target = target[gt_label != 0]

        return self.loss_reg(input, target)
