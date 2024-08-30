import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.Softmax()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, batch=False):
        hidden_1 = torch.relu(self.fc1(state))
        #can only do batch normalization if we're forwarding an entire batch, not on normal inputs
        # if batch:
        #     hidden_2 = torch.relu(self.bn(self.fc2(hidden_1)))
        # else:
        #     hidden_2 = torch.relu(self.fc2(hidden_1))
        hidden_2 = torch.relu(self.fc2(hidden_1))
        output = self.softmax(self.fc3(hidden_2))
        return output

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        # x = x.reshape((x.shape[0],x.shape[2]))
        # action = action.reshape((action.shape[0], action.shape[2]))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x