#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import os, requests
from typing import Any
from mediator import *
from utils import global_param


from abc import ABC, abstractmethod

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self, offline=True, soft=False, prefix=''):
        super().__init__()
        self.offline = offline
        self.soft = soft
        self.prompt_prefix = prefix
        self.plans_dict = {}
        self.mediator = None
        
        self.dialogue_system = ''               
        self.dialogue_user = ''
        self.dialogue_logger = ''     
        self.show_dialogue = False
        
        if not offline:
            self.llm_model = "vicuna-33b"
            self.llm_url = 'http://localhost:3300/v1/chat/completions'
            # self.llm_model = "chatglm_Turbo"
            # self.llm_url = 'http://10.109.116.3:6000/chat'
            self.plans_dict = {}
            if self.llm_model == "vicuna-33b":
                self.init_llm()
        
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
        self.mediator.reset()
        # if not self.offline:
        #     self.online_planning("reset")
        
    def init_llm(self):
        self.dialogue_system += self.prompt_prefix

        ## set system part
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': self.llm_model, "messages":[{"role": "system", "content": self.prompt_prefix}]}
                response = requests.post(self.llm_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to initialize: status code {response.status_code}"                
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to initialize: {e}")

    def query_codex(self, prompt_text):
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                #response =  openai.Completion.create(prompt_text)
                headers = {'Content-Type': 'application/json'}
                
                # print(f"user prompt:{prompt_text}")
                if self.llm_model == "chatglm_Turbo":
                    data = {'model': self.llm_model, "prompt":[{"role": "user", "content": self.prompt_prefix + prompt_text}]}     
                elif self.llm_model == "vicuna-33b":
                    data = {'model': self.llm_model, "messages":[{"role": "user", "content": prompt_text}]}
                response = requests.post(self.llm_url, headers=headers, json=data)

                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to query: status code {response.status_code}"
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to query: {e}")
                
        try:
            plan = re.search("Action[s]*\:\s*\{([\w\s\<\>\,]*)\}", result, re.I | re.M).group(1)
            return plan
        except:
            print(f"LLM response invalid format: '{result}'.")
            return self.query_codex(prompt_text)   
        
    def plan(self, text, n_ask=10):
        if text in self.plans_dict.keys():
            plans, probs = self.plans_dict[text]
        else:
            print(f"new obs: {text}")
            plans = {}
            for _ in range(n_ask):
                plan = self.query_codex(text)
                if plan in plans.keys():
                    plans[plan] += 1/n_ask
                else:
                    plans[plan] = 1/n_ask
            
            plans, probs = list(plans.keys()), list(plans.values())
            self.plans_dict[text] = (plans, probs)
            
            for k, v in self.plans_dict.items():
                print(f"{k}:{v}")

        return plans, probs
    
    def __call__(self, obs):
        # self.mediator.reset()
        text = self.mediator.RL2LLM(obs)
        plans, probs = self.plan(text)
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        
        return skill_list, probs
    
    

class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = SimpleDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>", "pick up <key>"], [0.98, 0.02]],
                "Agent sees <nothing>, holds <key>."     : [["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>", "go to <door>, pick up <handle>, use <key>"], [0.68, 0.22, 0.04, 0.02, 0.02, 0.02]],
                "Agent sees <door>, holds <key>."        : [["go to <door>, open <door> with <key>", "go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>"], [0.62, 0.3, 0.06, 0.02]],
                "Agent sees <key>, <door>, holds <nothing>." : [["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, go to <door>, use <key>", "go to <key>, pick up <key>, explore"], [0.84, 0.08, 0.04, 0.02, 0.02]]
}     
    

class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = ColoredDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>."       : [["explore"],[1]],
                "Agent sees <nothing>, holds <color1 key>."    : [["explore","go to east"], [0.94,0.06]],
                "Agent sees <color1 key>, holds <nothing>."    : [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 door>, holds <nothing>."   : [["explore"],[1.0]],
                "Agent sees <color1 door>, holds <color1 key>.": [["go to <color1 door>, open <color1 door>","open <color1 door>"],[0.72,0.28]],
                "Agent sees <color1 door>, holds <color2 key>.": [["explore", "go to <color2 key>"],[0.98,0.02]],
                "Agent sees <color1 key>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 key>, <color2 key>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.81,0.19]],
                "Agent sees <color1 key>, <color2 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.73,0.27]],
                "Agent sees <color1 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.84,0.16]],
                "Agent sees <color1 key>, <color1 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.79,0.21]],
                "Agent sees <color1 key>, <color2 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>", "go to <color2 door>, open <color2 door>"],[0.71,0.29]],
                "Agent sees <color1 key>, <color2 key>, <color2 door>, holds <nothing>.": [["go to <color2 key>, pick up <color2 key>","pick up <color2 key>","go to <color1 key>, pick up <color1 key>"],[0.72,0.24,0.04]],
                "Agent sees <color1 key>, <color2 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>"," pick up <color1 key>"],[0.94,0.06]],
}
        
    def plan(self, text):
        pattern= r'\b(blue|green|grey|purple|red|yellow)\b'
        color_words = re.findall(pattern, text)

        words = list(set(color_words))
        words.sort(key=color_words.index)
        color_words = words
        color_index =['color1','color2']
        if color_words != []:
            for i in range(len(color_words)):
                text = text.replace(color_words[i], color_index[i])

        plans, probs = super().plan(text)

        plans = str(plans)
        for i in range(len(color_words)):
            plans = plans.replace(color_index[i], color_words[i])
        plans = eval(plans)

        return plans, probs


class TwoDoor_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = TwoDoor_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door1>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <nothing>, holds <key>."     : [["explore"], [1.0]],
                "Agent sees <door1>, holds <key>."        : [["go to <door1>, open <door1>"], [1.0]],
                "Agent sees <key>, <door1>, holds <nothing>." : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, <door1>, <door2>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <key>.": [["go to <door1>, open <door1>", "go to <door2>, open <door2>"], [0.5, 0.5]],
            }  
                                                            
                                                            
def Planner(task, offline=True, soft=False, prefix=''):
    if task.lower() == "simpledoorkey":
        planner = SimpleDoorKey_Planner(offline, soft, prefix)
    elif task.lower() == "lavadoorkey":
        planner = SimpleDoorKey_Planner(offline, soft, prefix)
    elif task.lower() == "coloreddoorkey":
        planner = ColoredDoorKey_Planner(offline, soft, prefix)
    elif task.lower() == "twodoor":
        planner = TwoDoor_Planner(offline, soft, prefix)
    return planner
                                                            
                                                            