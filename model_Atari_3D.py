import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_channel, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_channel (int): Dimension of input state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input size: 4 x 84 x 84

        # conv layer 1:
        ## output size = (W-F+2P)/S +1 = (84-8)/4 +1 = 20
        # the output Tensor will have the dimensions: (32, 20, 20)
        self.conv1 = nn.Conv2d(input_channel, 32, 8, 4)

        # conv layer 2:
        ## output size = (W-F+2P)/S +1 = (20-4)/2 +1 = 9
        # the output Tensor will have the dimensions: (64, 9, 9)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)

        # conv layer 3:
        ## output size = (W-F+2P)/S +1 = (9-3)/1 +1 = 8
        # the output Tensor will have the dimensions: (64, 7, 7)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        
        # 64 outputs * the 7*7 filtered/pooled map size
        self.fc1 = nn.Linear(64*7*7, 512)
        
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state

        # three conv/relu layers
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        
        # final output
        return x
