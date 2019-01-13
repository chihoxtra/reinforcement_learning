import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Unity Network
"""

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
        self.duel = duel

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)

        self.fc2 = nn.Linear(64, 64) #-> common

        # 2 output channels (for the action_size classes)
        self.fc4a = nn.Linear(64, action_size)

        ####################################

        self.fc2v = nn.Linear(64, 64)

        # 2 output channels (for the 1 on/off classes)
        self.fc4v = nn.Linear(64, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state

        # one linear relu layer
        x = F.relu(self.fc1(x))

        # one linear relu layer for action
        common_out = F.relu(self.fc2(x))

        # one linear output layer for action
        a = self.fc4a(common_out)

        # if duel network is applied
        if self.duel:
            # for actions
            # one linear layer
            v = F.relu(self.fc2v(common_out))

            v = self.fc4v(v)

            a_adj = a - a.mean()

            out = v + a_adj
        else:
            out = a

        # final output
        return out
