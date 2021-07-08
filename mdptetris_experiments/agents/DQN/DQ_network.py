from numpy import isin
from torch import nn


class DQ_network(nn.Module):
    """
    Define a neural network for use in DQN. 
    """

    def __init__(self):
        super(DQ_network, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(6, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class DQN_2D(nn.Module):
    """
    Define a neural network for use with 2D Tetris state spaces.
    """
    
    def __init__(self, num_inputs):
        super(DQN_2D, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 1)