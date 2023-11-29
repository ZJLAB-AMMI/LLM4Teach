import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from .goto_goal import GoTo_Goal

class Pickup(BaseSkill):
    def __init__(self, init_obs, target_obj):
        self.unpack_obs(self.obs)
        self.target_obj = target_obj
        # get the coordinate of target_obj
    
    def __call__(self):
        if self.carrying == self.target_obj:
            return None, True, False
        else:
            # get the coordinate of target_obj
            target_pos = tuple(np.argwhere(self.map==self.target_obj)[0])
            print(target_pos)
            action = GoTo_Goal(obs, target_pos)()
            return action, False, False
        
        # fwd_pos = tuple(self.agent_pos + DIR_TO_VEC[self.agent_dir])
        # if fwd_pos == target_pos:
        #     return 3, False, False