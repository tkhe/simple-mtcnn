import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.roi_cls_head = nn.Conv2d(32, 2, kernel_size=1)
        self.roi_reg_head = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        roi_cls = self.roi_cls_head(x)
        roi_reg = self.roi_reg_head(x)
        return roi_cls, roi_reg
