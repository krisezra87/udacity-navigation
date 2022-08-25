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

        # Define the number of nodes in the network here.  Not changing
        # and keeps the signature simple
        net1_nodes = 64
        net2_nodes = 64

        # Build the Q network with 2 hidden layers between the state input and the action output
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net1 = nn.Linear(state_size,net1_nodes)
        self.net2 = nn.Linear(net1_nodes,net2_nodes)
        self.net3 = nn.Linear(net2_nodes,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # To get the mapping, we need to do a relaxed linear activation unit
        # (ReLU) through the network we have created.  It will provide linear
        # output when the result is greater than zero, otherwise it will give
        # zero output.  Note that we do NOT want relu activation for the output
        # layer... that cost more debug time than I would care to admit.
        x = F.relu(self.net1(state))
        x = F.relu(self.net2(x))
        x = self.net3(x)
        return x
