from torch import nn

class DQ_network(nn.Module):
    """
    Define a neural netowrk for Tetris
    """
    def __init__(self):
        super(DQ_network).__init__()

        self.conv1 = nn.Sequential(nn.Linear())

    def forward(self, x):
        pass