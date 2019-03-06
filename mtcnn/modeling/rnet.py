import torch.nn as nn


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.roi_cls_head = nn.Linear(128, 2)
        self.roi_reg_head = nn.Linear(128, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        roi_cls = self.roi_cls_head(x)
        roi_reg = self.roi_reg_head(x)
        return roi_cls, roi_reg
