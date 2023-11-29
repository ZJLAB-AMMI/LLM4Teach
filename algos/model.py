#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   student_net.py
@Time    :   2023/07/14 16:34:11
@Author  :   Zhou Zihao 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np


class NNBase(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        width, height, channel = obs_space["image"]
    
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(channel, 16, (3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU()
        )
        
        dummy_x = torch.zeros((1, channel, height, width))
        embedding_size = np.prod(self.image_conv(dummy_x).shape)
        return embedding_size, action_space
    
    def forward(self, obs, masks=None, states=None):
        raise NotImplementedError

        
class MLPBase(NNBase):
    def __init__(self, obs_space, action_space):
        embedding_size, action_space = super().__init__(obs_space, action_space)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_size, 64),
        #     nn.Tanh()
        # )
        
        # # Define actor's model
        # self.actor = nn.Linear(64, action_space)
        # # Define critic's model
        # self.critic = nn.Linear(64, 1)
        
        # # Define actor's model
        # self.actor = nn.Sequential(
        #     nn.Linear(64, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, action_space)
        # )
        # # Define critic's model
        # self.critic = nn.Sequential(
        #     nn.Linear(64, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 1)
        # )
        
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def init_states(self, device=None, num_trajs=1):
        return None

    def forward(self, obs, masks=None, states=None):
        input_dim = len(obs.size())
        assert input_dim == 4, "observation dimension expected to be 4, but got {}.".format(input_dim)
        
        # feature extractor
        x = obs.transpose(1, 3) # [num_trans, channels, height, width]
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1) # [num_trans, -1]
        embedding = x
        # embedding = self.fc(x)

        # actor-critic
        value = self.critic(embedding).squeeze(1)
        action_logits = self.actor(embedding)
        dist = Categorical(logits=action_logits)

        return dist, value, embedding
        

class LSTMBase(NNBase):
    def __init__(self, obs_space, action_space):
        embedding_size, action_space = super().__init__(obs_space, action_space)
    
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU()
        )
        self.core = nn.LSTM(256, 256, 2)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def init_states(self, device, num_trajs=1):
        return (torch.zeros(self.core.num_layers, num_trajs, self.core.hidden_size).to(device),
                torch.zeros(self.core.num_layers, num_trajs, self.core.hidden_size).to(device))

    def forward(self, obs, masks, states):
        input_dim = len(obs.size())
        if input_dim == 4:
            unroll_length = obs.shape[0]
            num_trajs = 1
        elif len(obs.size()) == 5:
            unroll_length, num_trajs, *_ = obs.shape
            obs = torch.flatten(obs, 0, 1) # [unroll_length * num_trajs, width, height, channels]
        else:
            assert False, "observation dimension expected to be 4 or 5, but got {}.".format(input_dim)

        # feature extractor
        x = obs.transpose(1, 3) # [unroll_length * num_trajs, channels, height, width]
        x = self.image_conv(x)
        x = x.reshape(unroll_length * num_trajs, -1) # [unroll_length * num_trajs, -1]
        x = self.fc(x)
        
        # LSTM
        core_input = x.view(unroll_length, num_trajs, -1) # [unroll_length, num_trajs, -1] 
        masks = masks.view(unroll_length, 1, num_trajs, 1) # [unroll_length, 1, num_trajs, 1]
        core_output_list = []
        for inp, mask in zip(core_input.unbind(), masks.unbind()):
            states = tuple(mask * s for s in states)
            output, states = self.core(inp.unsqueeze(0), states)
            core_output_list.append(output)
        core_output = torch.cat(core_output_list) # [unroll_length, num_trajs, -1]
        core_output = core_output.view(unroll_length * num_trajs, -1) # [unroll_length * num_trajs, -1]

        # actor-critic
        action_logits = self.actor(core_output)
        dist = Categorical(logits=action_logits)
        value = self.critic(core_output).squeeze(1)

        return dist, value, states
