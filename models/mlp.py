import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from models.hyper_q_network import Hyper_Critic
from models.hyper_q_network_random import Hyper_Critic_Random
from models.hyper_network import Meta_Embadding
from models.hyper_network import ResBlock


def weight_init(module):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(module.weight, -w, w)
        nn.init.uniform_(module.bias, -w, w)


WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4
N1 = 256;
N2 = 256


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        h_size = 256
        d2rl = False
        state_features = False
        res_35 = False

        if d2rl:
            self.q1 = D2rl_Q(state_dim, action_dim, h_size)
            self.q2 = D2rl_Q(state_dim, action_dim, h_size)

        elif res_35:
            self.q1 = Super_Critic(state_dim, action_dim, h_size)
            self.q2 = Super_Critic(state_dim, action_dim, h_size)

        elif state_features:
            self.q1 = Deep_Critic(state_dim, action_dim, h_size)
            self.q2 = Deep_Critic(state_dim, action_dim, h_size)

        else:
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, h_size),
                nn.ReLU(),
                nn.Linear(h_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 1)
            )
            # Q2 architecture
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, h_size),
                nn.ReLU(),
                nn.Linear(h_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 1)
            )

    def forward(self, state, action, logger=None):
        if logger is not None:
            logger['w1'][-1].append(self.l1.weight.detach().cpu().numpy())
            logger['w2'][-1].append(self.l2.weight.detach().cpu().numpy())
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def Q1(self, state, action, logger=None):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        return q1


class D2rl_Q(nn.Module):
    def __init__(self, state_dim, action_dim, h_size):
        super(D2rl_Q, self).__init__()

        in_dim = state_dim + action_dim + h_size
        self.l1_1 = nn.Linear(state_dim + action_dim, h_size)
        self.l1_2 = nn.Linear(in_dim, h_size)
        self.l1_3 = nn.Linear(in_dim, h_size)
        self.l1_4 = nn.Linear(in_dim, h_size)
        self.out1 = nn.Linear(h_size, 1)

    def forward(self, xu):
        x1 = F.relu(self.l1_1(xu))
        x1 = torch.cat([x1, xu], dim=1)
        x1 = F.relu(self.l1_2(x1))
        x1 = torch.cat([x1, xu], dim=1)
        x1 = F.relu(self.l1_3(x1))
        x1 = torch.cat([x1, xu], dim=1)
        x1 = F.relu(self.l1_4(x1))
        x1 = self.out1(x1)
        return x1


# res35 critic
class Super_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Super_Critic, self).__init__()
        h_size = 1024
        n_blocks = 35

        # Q1 architecture
        self.q1_emb = nn.Sequential(nn.Linear(state_dim, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    nn.Linear(h_size, state_dim)
                                    )

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2_emb = nn.Sequential(nn.Linear(state_dim, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    ResBlock(h_size, h_size),
                                    nn.Linear(h_size, state_dim)
                                    )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def init_layers(self):

        for module in self.q1_emb.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)
        for module in self.q2_emb.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def forward(self, state, action):
        z1 = self.q1_emb(state)
        z2 = self.q2_emb(state)
        sa1 = torch.cat([z1, action], 1)
        sa2 = torch.cat([z2, action], 1)
        q1 = self.q1(sa1)
        q2 = self.q2(sa2)
        return q1, q2

    def Q1(self, state, action, logger):
        z1 = self.q1_emb(state)
        sa1 = torch.cat([z1, action], 1)
        q1 = self.q1(sa1)
        return q1


# state_features critic
class Deep_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Deep_Critic, self).__init__()
        h_size = 256
        n_blocks = 70
        # Q1 architecture
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, h_size), )
        for i in range(n_blocks):
            self.q1.add_module("block_{}".format(i), ResBlock(h_size, h_size))
        self.q1.add_module("relu_1", nn.ReLU())
        self.q1.add_module("fc", nn.Linear(h_size, 1))

        # Q2 architecture
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, h_size), )
        for i in range(n_blocks):
            self.q2.add_module("block_{}".format(i), ResBlock(h_size, h_size))
        self.q2.add_module("relu_1", nn.ReLU())
        self.q2.add_module("fc", nn.Linear(h_size, 1))

    def forward(self, state, action):
        sa1 = torch.cat([state, action], 1)
        sa2 = torch.cat([state, action], 1)
        q1 = self.q1(sa1)
        q2 = self.q2(sa2)
        return q1, q2

    def Q1(self, state, action, logger):
        sa1 = torch.cat([state, action], 1)
        q1 = self.q1(sa1)
        return q1
