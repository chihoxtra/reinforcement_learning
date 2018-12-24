import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        self.fc1 = nn.Linear(state_size, 1024)

        self.fc2 = nn.Linear(1024, 128)
        #self.fc3 = nn.Linear(256, 256)
        #self.fc4 = nn.Linear(256, 128)
        #self.fc5 = nn.Linear(128, 64)

        # 2 output channels (for the 4 classes)
        self.fc6 = nn.Linear(128, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        #print("forwarded", x.shape)

        # one linear layer
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)

        # one linear layer
        x = F.relu(self.fc2(x))

        # one linear layer
        #x = F.relu(self.fc3(x))

        # one linear layer
        #x = F.relu(self.fc4(x))

        # one linear layer
        #x = F.relu(self.fc5(x))

        # one linear layer
        x = self.fc6(x)

        # a softmax layer to convert the outputs into a distribution of action scores
        #x = F.softmax(x, dim=-1)

        # final output
        return x
