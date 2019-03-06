import torch
import torch.nn as nn

from mtcnn.config import cfg


class Loss(object):
    def __init__(self):
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reg = nn.SmoothL1Loss()

    def __call__(self, cls, reg, label, target):
        cls_loss = self.cls_loss(cls, label)
        reg_loss = self.reg_loss(reg, target, label)
        return cfg.TRAIN.LOSS_RATIO[0] * cls_loss + \
               cfg.TRAIN.LOSS_RATIO[1] * reg_loss

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
