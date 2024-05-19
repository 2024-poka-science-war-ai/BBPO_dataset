"""
The file of actor and critic architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from parameters import TAU


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, a_dim)
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(256)
        self.bn1d_3 = nn.BatchNorm1d(128)
        self.a_dim = a_dim
        self.activ = nn.ELU()

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            mu and sigma of policy considering current observation
        """
        s1, s2 = s

        s1 = self.bn1d_1(self.activ(self.fc1(s1)))
        s1 = self.bn1d_2(self.activ(self.fc2(s1)))
        s1 = self.bn1d_3(self.activ(self.fc3(s1)))
        return self.fc4(s1)

    def choose_action(self, s):
        """
        Choose action by normal distribution

        Args:
            s: current observation
        Returns:
            action tensor sampled from policy(normal distribution),
            log probability of the action
        """
        with torch.no_grad():
            self.eval()
            action_prob_ts = self.forward(s)
            action_prob_ts = torch.softmax(action_prob_ts, dim=1)
            action_prob_np = action_prob_ts.squeeze().cpu().numpy()
        return action_prob_np


class Critic(nn.Module):
    """
    Critic network
    """

    def __init__(self, s_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(256)
        self.bn1d_3 = nn.BatchNorm1d(128)
        self.activ = nn.ELU()

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            estimated value of current state
        """
        s1, s2 = s

        s1 = self.bn1d_1(self.activ(self.fc1(s1)))
        s1 = self.bn1d_2(self.activ(self.fc2(s1)))
        s1 = self.bn1d_3(self.activ(self.fc3(s1)))
        return self.fc4(s1)
