import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from planner import Planner
from skill import GoTo_Goal, Explore, Pickup, Drop, Toggle, Wait
from mediator import IDX_TO_SKILL, IDX_TO_OBJECT

# single step (can handle soft planner)
class TeacherPolicy():
    def __init__(self, task, offline, soft, prefix, action_space, agent_view_size):
        self.planner = Planner(task, offline, soft, prefix)
        self.agent_view_size = agent_view_size
        self.action_space = action_space
        
    def get_skill_name(self, skill):
        try:
            return IDX_TO_SKILL[skill["action"]] + " " + IDX_TO_OBJECT[skill["object"]]
        except AttributeError:
            return "None"
        
    def reset(self):
        self.skill = None
        self.skill_list = []
        self.skill_teminated = False
        self.planner.reset() 

    def skill2teacher(self, skill):
        skill_action = skill['action']
        if skill_action == 0:
            teacher = Explore(self.agent_view_size)
        elif skill_action == 1:
            teacher = GoTo_Goal(skill['coordinate'])
        elif skill_action == 2:
            teacher = Pickup(skill['object'])
        elif skill_action == 3:
            teacher = Drop(skill['object'])
        elif skill_action == 4:
            teacher = Toggle(skill['object'])
        elif skill_action == 6:
            teacher = Wait()
        else:
            assert False, "invalid skill"
        return teacher
    
    def get_action(self, skill_list, obs):
        teminated = True
        action = None
        while not action and teminated and len(skill_list) > 0:
            skill = skill_list.pop(0)
            teacher = self.skill2teacher(skill)
            action, teminated = teacher(obs)
                
        if action == None:

            action = 6
            
        action = np.array([i == action for i in range(self.action_space)], dtype=np.float32)
            
        return action
    
    def __call__(self, obs):
        skill_list, probs = self.planner(obs)
        action = np.zeros(self.action_space)
        for skills, prob in zip(skill_list, probs):
            action += self.get_action(skills, obs) * prob
        return action


# class TeacherPolicy():
#     def __init__(self, task, ideal, seed, agent_view_size):
#         self.planner = Planner(task, ideal, seed)
#         self.agent_view_size = agent_view_size
        
#     @property
#     def current_skill(self):
#         try:
#             return IDX_TO_SKILL[self.skill["action"]] + " " + IDX_TO_OBJECT[self.skill["object"]]
#         except AttributeError:
#             return "None"
        
#     def reset(self):
#         self.skill = None
#         self.skill_list = []
#         self.skill_teminated = False
#         self.skill_truncated = False
#         self.planner.reset() 
        
#     def initial_planning(self, decription, task_example):
#         self.planner.initial_planning(decription, task_example)

#     def skill2teacher(self, obs):
#         skill_action = self.skill['action']
#         if skill_action == 0:
#             teacher = Explore(obs, self.agent_view_size)
#         elif skill_action == 1:
#             teacher = GoTo_Goal(obs, self.skill['coordinate'])
#         elif skill_action == 2:
#             teacher = Pickup(obs, self.skill['object'])
#         elif skill_action == 3:
#             teacher = Drop(obs, self.skill['object'])
#         elif skill_action == 4:
#             teacher = Toggle(obs, self.skill['object'])
#         else:
#             assert False, "invalid skill"
#         return teacher
    
#     def switch_skill(self, obs):
#         if self.skill_truncated or len(self.skill_list) == 0:
#             self.skill_list = self.planner(obs) # ask LLM
#             self.can_truncated = False
#         self.skill = self.skill_list.pop(0)
    
#     def __call__(self, obs, max_tries=5, always_ask=True):
#         if always_ask:
#             self.skill_truncated = True
#         action = None
#         tries = 0
#         self.can_truncate = True
#         while not action and tries <= max_tries:
#             if self.skill_teminated or self.skill_truncated:
#                 self.switch_skill(obs)
#             teacher = self.skill2teacher(obs)
#             action, self.skill_teminated, self.skill_truncated = teacher(self.can_truncated)
#             tries += 1
                
#         if action == None:
#             print(obs[:, :, 0])
#             print(obs[:, :, 2])
#             print(obs[:, :, 3])
#             print(teacher.message)
#             assert False, "teacher {} cannot give an action".format(self.current_skill)
            
#         return action


