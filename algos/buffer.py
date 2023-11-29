#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   buffer.py
@Time    :   2023/04/19 09:40:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch


# def Merge_Buffers(buffers, device='cpu'):
#     merged = Buffer(device=device)
#     for buf in buffers:
#         offset = len(merged)

#         merged.obs  += buf.obs
#         merged.actions += buf.actions
#         merged.rewards += buf.rewards
#         merged.values  += buf.values
#         merged.returns += buf.returns
#         merged.log_probs += buf.log_probs
#         merged.teacher_probs += buf.teacher_probs

#         merged.ep_returns += buf.ep_returns
#         merged.ep_lens    += buf.ep_lens


#         merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
#         merged.ptr += buf.ptr

#     return merged


class Buffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    """
    def __init__(self, gamma=0.99, lam=0.95, device='cpu'):
        self.gamma = gamma
        self.lam = lam # unused
        self.device = device

    def __len__(self):
        return self.ptr
    
    def clear(self):
        self.obs  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.log_probs = []
        self.teacher_probs = []

        self.ptr = 0
        self.traj_idx = [0]
        self.returns = []
        self.ep_returns = [] # for logging
        self.ep_lens    = []

    def store(self, state, action, reward, value, log_probs, teacher_probs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        # print(obs.shape, action.shape, reward.shape, value.shape, log_probs.shape)
        self.obs  += [state.squeeze(0)]
        self.actions += [action.squeeze()]
        self.rewards += [reward.squeeze()]
        self.values  += [value.squeeze()]
        self.log_probs += [log_probs.squeeze()]
        self.teacher_probs += [teacher_probs]
        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []
        R = last_val
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) 

        self.returns += returns
        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]
    
    def get(self):
        return(
            np.array(self.obs),
            np.array(self.actions),
            np.array(self.returns),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.teacher_probs)
        )

    def sample(self, batch_size=64, recurrent=False):
        if recurrent:
            random_indices = np.random.permutation(len(self.ep_lens))
            last_index = random_indices[-1]
            sampler = []
            indices = []
            num_sample = 0
            for i in random_indices:
                indices.append(i)
                num_sample += self.ep_lens[i]
                if num_sample > batch_size or i == last_index:
                    sampler.append(indices)
                    indices = []
                    num_sample = 0
            # random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            # sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        else:
            random_indices = SubsetRandomSampler(range(self.ptr))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

        observations, actions, returns, values, log_probs, teacher_probs = map(torch.Tensor, self.get())

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for indices in sampler:
            if recurrent:              
                obs_batch       = [observations[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                action_batch    = [actions[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                return_batch    = [returns[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                advantage_batch = [advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                values_batch    = [values[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                mask            = [torch.ones_like(r) for r in return_batch]
                log_prob_batch  = [log_probs[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                teacher_prob_batch  = [teacher_probs[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]

                obs_batch       = pad_sequence(obs_batch, batch_first=False) # [unroll_length, num_trajs, ...]
                action_batch    = pad_sequence(action_batch, batch_first=False).flatten(0,1)
                return_batch    = pad_sequence(return_batch, batch_first=False).flatten(0,1)
                advantage_batch = pad_sequence(advantage_batch, batch_first=False).flatten(0,1)
                values_batch    = pad_sequence(values_batch, batch_first=False).flatten(0,1)
                mask            = pad_sequence(mask, batch_first=False).flatten(0,1)
                log_prob_batch  = pad_sequence(log_prob_batch, batch_first=False).flatten(0,1)
                teacher_prob_batch = pad_sequence(teacher_prob_batch, batch_first=False).flatten(0,1)
            else:
                obs_batch       = observations[indices]
                action_batch    = actions[indices]
                return_batch    = returns[indices]
                advantage_batch = advantages[indices]
                values_batch    = values[indices]
                mask            = torch.FloatTensor([1])
                log_prob_batch  = log_probs[indices]
                teacher_prob_batch = teacher_probs[indices]


            yield obs_batch.to(self.device), action_batch.to(self.device), return_batch.to(self.device), advantage_batch.to(self.device), values_batch.to(self.device), mask.to(self.device), log_prob_batch.to(self.device), teacher_prob_batch.to(self.device)