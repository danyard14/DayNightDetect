import torch.nn as nn
from torch.nn import Module


class DayNightDetector(Module):
    def __init__(self):
        super(DayNightDetector, self).__init__()
        # (in channels should be the dimension of the channels of the images we learn (rgb: 3, gray: 1)) self.conv1 = nn.Conv1d(in_channels=)
