from torch import nn
import torch.nn.functional as F


class NNHeuristicAction(nn.Module):
    """
    Define a neural network for use in DQN with a heuristic input. 
    """

    def __init__(self, input_dims, output_dims):
        super(NNHeuristicAction, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dims, 64))
        self.conv2 = nn.Sequential(nn.Linear(64, 64))
        self.conv3 = nn.Sequential(nn.Linear(64, output_dims))

    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class NN1DAction(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(NN1DAction, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dims, 256))
        self.conv2 = nn.Sequential(nn.Linear(256, 256))
        self.conv3 = nn.Sequential(nn.Linear(256, output_dims))
    
    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class PPONN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(PPONN, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(input_dims, 256))
        self.conv2 = nn.Sequential(nn.Linear(256, 256))
        self.conv3 = nn.Sequential(nn.Linear(256, 256))
        self.actor = nn.Linear(256, output_dims)
        self.critic = nn.Linear(256, 1)
        self._initialise_weights()

    def _initialise_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.actor(x), self.critic(x)