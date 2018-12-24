import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_channel, action_size, seed):
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

        # input size: 4 x 84 x 84

        # conv layer 1:
        ## output size = (W-F+2P)/S +1 = (84-5)/1 +1 = 80
        # the output Tensor will have the dimensions: (8, 80, 80)
        self.conv1 = nn.Conv2d(input_channel, 16, 5)

        # maxpool layer 1:
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (16, 40, 40)
        self.pool1 = nn.MaxPool2d(2, 2)

        # conv layer 2:
        ## output size = (W-F+2P)/S +1 = (40-3)/1 +1 = 38
        # the output Tensor will have the dimensions: (32, 38, 38)
        self.conv2 = nn.Conv2d(8, 32, 3)

        # maxpool layer 2:
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (16, 19, 19)
        self.pool2 = nn.MaxPool2d(2, 2)

        # conv layer 3:
        ## output size = (W-F+2P)/S +1 = (19-3)/1 +1 = 17
        # the output Tensor will have the dimensions: (64, 17, 17)
        #self.conv3 = nn.Conv2d(16, 16, 3)

        # maxpool layer 3:
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (16, 8, 8) #round down
        #self.pool3 = nn.MaxPool2d(2, 2)

        # 16 outputs * the 19*19 filtered/pooled map size
        # 256 output channels
        self.fc1 = nn.Linear(16*19*19, 1024)

        self.fc2 = nn.Linear(1024, 64)

        # 2 output channels (for the 4 classes)
        self.fc6 = nn.Linear(64, action_size)

        self.dropout1 = nn.Dropout(p=0.2)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state

        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.dropout1(x)

        #x = self.pool3(F.relu(self.conv3(x)))
        #x = self.dropout1(x)

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)

        # one linear layer
        x = F.relu(self.fc2(x))

        # one linear layer
        x = self.fc6(x)

        # a softmax layer to convert the outputs into a distribution of action scores
        #x = F.softmax(x, dim=-1)

        # final output
        return x
