import torch
from torch import nn


class CLS_end(nn.Module):
    def __init__(self, backbone_in, num_cls):
        super(CLS_end, self).__init__()
        self.backbone = backbone_in
        self.fc = nn.Linear(1000, num_cls)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
