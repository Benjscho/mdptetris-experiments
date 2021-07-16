from torch import nn
import torch.nn.functional as F


class NNHeuristic(nn.Module):
    """
    Define a neural network for use in DQN with a heuristic input. 
    """

    def __init__(self, input_dims: int=6):
        super(NNHeuristic, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dims, 64))
        self.conv2 = nn.Sequential(nn.Linear(64, 64))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class NN1D(nn.Module):
    def __init__(self, input_dims=200):
        super(NN1D, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dims, 256))
        self.conv2 = nn.Sequential(nn.Linear(256, 256))
        self.conv3 = nn.Sequential(nn.Linear(256, 1))
    
    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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

    def _initialise_weights(self):
        pass