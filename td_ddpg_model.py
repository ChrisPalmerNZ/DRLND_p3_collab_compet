import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, max_action, fc1_units=400, fc2_units=300, 
                 leakiness=0.01):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.max_action = float(max_action)
        self.leakiness = leakiness

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)    

    def forward(self, state):    
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        # Use leaky relu if specified
        if self.leakiness > 0:
            x = F.leaky_relu(x, negative_slope=self.leakiness)
            x = F.leaky_relu(self.fc2(x), negative_slope=self.leakiness)            
        else:    
            x = F.relu(x)
            x = F.relu(self.fc2(x))

        # multiply by max_action to get back to original dimensionality...
        x = self.max_action * torch.tanh(self.fc3(x))

        return x

class Critic(nn.Module):
    """Critic (Q values) Model."""

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300, 
                 leakiness=0.01):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        # First Critic neural network
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        # Second Critic neural network
        self.fc4 = nn.Linear(state_size + action_size, fc1_units)
        self.fc5 = nn.Linear(fc1_units, fc2_units)
        self.fc6 = nn.Linear(fc2_units, 1)
        
        self.leakiness = leakiness

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)    
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)   

    def forward(self, state, action_space):    
        """Forward propogate signal through both NN of Critic"""
        state_action = torch.cat([state, action_space], 1)

        # Forward on first Critic NN 
        x1 = self.fc1(state_action)
        # Use leaky relu if specified
        if self.leakiness > 0:
            x1 = F.leaky_relu(x1, negative_slope=self.leakiness)
            x1 = F.leaky_relu(self.fc2(x1), negative_slope=self.leakiness)            
        else:    
            x1 = F.relu(x1)
            x1 = F.relu(self.fc2(x1))

        x1 = self.fc3(x1)

        # Forward on second Critic NN 
        x2 = self.fc4(state_action)
        if self.leakiness > 0:
            x2 = F.leaky_relu(x2, negative_slope=self.leakiness)
            x2 = F.leaky_relu(self.fc5(x2), negative_slope=self.leakiness)            
        else:  
            x2 = F.relu(x2)
            x2 = F.relu(self.fc5(x2))
            
        x2 = self.fc6(x2)

        return x1, x2

    def Q1(self, state, action_space):    
        """Forward propogate signal of first Critic for just first Q value
           Used for gradient ascent when updating weights of Actor model
        """
        state_action = torch.cat([state, action_space], 1)
        x1 = self.fc1(state_action)
        if self.leakiness > 0:
            x1 = F.leaky_relu(x1, negative_slope=self.leakiness)
            x1 = F.leaky_relu(self.fc2(x1), negative_slope=self.leakiness)            
        else:    
            x1 = F.relu(x1)
            x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1      
        