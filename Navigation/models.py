import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Simple Actor (Policy) Model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units(int): number of nodes in first fully connected hidden layer
            fc2_units(int): number of nodes in second fully connected hidden layer
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Defining the hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Output layer
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Build network that maps state to action values.
        """
        
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    
class DuelingQNetwork(nn.Module):
    """
    Actor (Policy) Model with Dueling Architecture
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units=64):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units(int): number of nodes in first fully connected hidden layer
            fc2_units(int): number of nodes in second fully connected hidden layer
            fc3_units(int): number of nodes in third fully connected hidden layer
        """
        
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Defining the hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.fc3_adv = nn.Linear(fc2_units, fc3_units)
        self.fc3_val = nn.Linear(fc2_units, fc3_units)
        
        # Output layer
        self.fc4_adv = nn.Linear(fc3_units, action_size)
        self.fc4_val = nn.Linear(fc3_units, 1)
        

    def forward(self, state):
        """
        Build network that maps state to action values.
        """
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.fc3_adv(x))
        val = F.relu(self.fc3_val(x))
        
        adv = self.fc4_adv(adv)
        val = self.fc4_val(val).expand(x.size(0), self.action_size)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x
