import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.core.constants import DIR_TO_VEC
from .base_skill import BaseSkill 


class Explore(BaseSkill):
    def __init__(self, obs, agent_view_size):
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size
        self.unpack_obs(obs) 
        self.get_room_boundary()
        self.action = None
        self.message = "none"
            
    def get_room_boundary(self):
        width = self.map.shape[0]
        height = self.map.shape[1]
        self.botX, self.botY = width, height
        for i in range(1, width-1):
            for j in range(1, height-1):
                if self.map[i, j] not in (2, 4):
                    pass
                elif self.botX == width and self.map[i, j + 1] in (2, 4):
                    self.botX = i + 1
                elif self.botY == height and self.map[i + 1, j] in (2, 4):
                    self.botY = j + 1
                else:
                    pass
                if self.botX != width and self.botY != height:
                    break
            
    def get_view(self, agent_dir, agent_pos=None):
        agent_pos = agent_pos if agent_pos else self.agent_pos
            
        # Facing right
        if agent_dir == 0:
            topX = agent_pos[0]
            topY = agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif agent_dir == 1:
            topX = agent_pos[0] - self.agent_view_size // 2
            topY = agent_pos[1]
        # Facing left
        elif agent_dir == 2:
            topX = agent_pos[0] - self.agent_view_size + 1
            topY = agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif agent_dir == 3:
            topX = agent_pos[0] - self.agent_view_size // 2
            topY = agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"
            
        # clip by room boundary
        topX = max(0, topX)
        topY = max(0, topY)
        botX = min(topX + self.agent_view_size, self.botX)
        botY = min(topY + self.agent_view_size, self.botY) 
        # print("[{}:{}, {}:{}]".format(topX, botX, topY, botY))

        return self.map[topX:botX, topY:botY]
            
    def get_grid_slice(self, agent_dir, agent_pos=None):
        agent_pos = agent_pos if agent_pos else self.agent_pos
        topX = 0
        topY = 0
        botX = self.botX
        botY = self.botY
        
        # Facing right
        if agent_dir == 0:
            topX = agent_pos[0] + self.agent_view_size // 2 + 1
        # Facing down
        elif agent_dir == 1:
            topY = agent_pos[1] + self.agent_view_size // 2 + 1
        # Facing left
        elif agent_dir == 2:
            botX = agent_pos[0] - self.agent_view_size // 2
        # Facing up
        elif agent_dir == 3:
            botY = agent_pos[1] - self.agent_view_size // 2
        else:
            assert False, "invalid agent direction"
        # print("[{}:{}, {}:{}]".format(topX, botX, topY, botY))
              
        return self.map[topX:botX, topY:botY]
    
    def object_in_sight(self, agent_dir, agent_pos=None):
        grid = self.get_view(agent_dir, agent_pos)
        for i in np.nditer(grid):
            # if i in (5, 6, 7, 9): # key, ball, box or lava 
            if i in (5, 6, 7): # key, ball, box
                return True
        return False
    
    def object_forward(self, agent_dir, agent_pos=None):
        x, y = self.agent_pos + DIR_TO_VEC[agent_dir]
        fwd_obj = self.map[x, y]
        if fwd_obj in (2, 4): # wall, door
            return 1
        # elif fwd_obj in (5, 6, 7, 9): # key, ball, box or lava 
        elif fwd_obj in (5, 6, 7): # key, ball, box or lava 
            return 2
        else:
            return 0
                    
    def count_unseen_grid(self, agent_dir, agent_pos=None):
        grid = self.get_grid_slice(agent_dir, agent_pos)
        if grid.size == 0:
            # print("Wall ahead in dir {}".format(agent_dir))
            return 0
        else:
            return np.count_nonzero(grid == 0)
        
    def __call__(self, can_truncate):
        # object in view?
        if self.object_in_sight(self.agent_dir):
            if can_truncate:
                self.message = "object in sight"
                return None, True, False # ask LLM
            elif self.object_forward(self.agent_dir) == 2: # avoid object
                if self.object_forward((self.agent_dir - 1) % 4) in (1,2):
                    return 1, False, False
                else:
                    return 0, False, False

        terminated = False
        truncated = False
        # unseen grid in forward direction?
        if self.count_unseen_grid(self.agent_dir) > 0:
            action = 2
        # unseen grid in leftward direction?
        elif self.count_unseen_grid((self.agent_dir - 1) % 4) > 0:
            action = 0
        # unseen grid in rightward direction?
        elif self.count_unseen_grid((self.agent_dir + 1) % 4) > 0:
            action = 1
        # unseen grid in backward direction?
        elif self.count_unseen_grid((self.agent_dir + 2) % 4, tuple(self.agent_pos + DIR_TO_VEC[self.agent_dir])) > 0:
            action = 0 # or 1
        # no unseen grid
        else:
            action = None
            terminated = True
            self.message = "no unseen grid"
            
        return action, terminated, truncated
    
