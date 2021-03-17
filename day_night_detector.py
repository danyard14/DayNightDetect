import torch.nn as nn
from torch.nn import Module, functional as F


class DayNightDetector(Module):
    def __init__(self):
        super(DayNightDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        self.fc1 = nn.Linear(in_features=12 * 43 * 43, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=2)
        # (in channels should be the dimension of the channels of the images we learn (rgb: 3, gray: 1)) self.conv1 = nn.Conv1d(in_channels=)

    def forward(self, t):
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=4, stride=4)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 43 * 43)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)

        t = F.softmax(t, dim=1)
        return t
