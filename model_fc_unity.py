import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, duel=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 128)

        self.fc2 = nn.Linear(128, 64)

        #self.fc3 = nn.Linear(64, 64)

        # 2 output channels (for the 4 classes)
        self.fc6 = nn.Linear(64, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state

        # one linear relu layer
        x = F.relu(self.fc1(x))

        # one linear relu layer
        x = F.relu(self.fc2(x))

        # one linear relu layer
        #x = F.relu(self.fc3(x))

        # one linear output ayer
        x = self.fc6(x)

        # final output
        return x
