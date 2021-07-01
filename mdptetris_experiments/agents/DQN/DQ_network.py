from numpy import isin
from torch import nn
import gym.spaces

class DQ_network(nn.Module):
    """
    Define a neural network for use in DQN. 
    """
    def __init__(self):
        super(DQ_network).__init__()

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