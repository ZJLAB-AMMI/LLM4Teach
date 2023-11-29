# [Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373)

## Abstract 
Recent studies have shown that Large Language Models (LLMs) can be utilized for solving complex sequential decision-making tasks by providing high-level instructions. However, LLM-based agents face limitations in real-time dynamic environments due to their lack of specialization in solving specific target problems. Moreover, the deployment of such LLM-based agents is both costly and time-consuming in practical scenarios. In this paper, we introduce a novel framework that addresses these challenges by training a smaller scale specialized student agent using instructions from an LLM-based teacher agent. By leveraging guided actions provided by the teachers, the prior knowledge of the LLM is distilled into the local student model. Consequently, the student agent can be trained with significantly less data. Furthermore, subsequent training with environment feedback empowers the student agents to surpass the capabilities of their teachers. We conducted experiments on three challenging MiniGrid environments to evaluate the effectiveness of our framework. The results demonstrate that our approach enhances sample efficiency and achieves superior performance compared to baseline methods.

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, [Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373).


## Running experiments

### Basics
Any algorithm can be run from the main.py entry point.

to train on a SimpleDoorKey environment,

```bash
python main.py train --task SimpleDoorKey --savedir train
```

to eval the trained model,

```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval
```

to train with given query result from LLM as teacher,

```bash
python main.py train --task SimpleDoorKey --savedir train --offline_planner
```

to eval teacher policy,
```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval --eval_teacher
```
## Local LLM and API
Please follow the instruction from [FastChat](https://github.com/lm-sys/FastChat) to install Vicuna model on local sever
Here are the commands to launch the API in terminal: 

Launch the controller
```bash
python3 -m fastchat.serve.controller --host localhost --port <controller_port>
```

to launch the model worker
```bash
python3 -m fastchat.serve.controller --host localhost --port <controller_port>
```

to launch the API
```bash
python3 -m fastchat.serve.controller --host localhost --port <controller_port>
```


## Logging details 
Tensorboard logging is enabled by default for all algorithms. The logger expects that you supply an argument named ```logdir```, containing the root directory you want to store your logfiles

The resulting directory tree would look something like this:
```
log/                         # directory with all of the saved models and tensorboard 
└── ppo                                 # algorithm name
    └── simpledoorkey                   # environment name
        └── save_name                   # unique save name 
            ├── acmodel.pt              # actor and critic network for algo
            ├── events.out.tfevents     # tensorboard binary file
            └── config.json             # readable hyperparameters for this run
```

Using tensorboard makes it easy to compare experiments and resume training later on.

To see live training progress

Run ```$ tensorboard --logdir=log``` then navigate to ```http://localhost:6006/``` in your browser

## Citation
If you find [our work](https://arxiv.org/abs/2311.13373) useful, please kindly cite: 
```bibtex
@article{zhou2023large,
  title={Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents},
  author={Zhou, Zihao and Hu, Bin and Zhang, Pu and Zhao, Chenyang and Liu, Bin},
  journal={arXiv preprint arXiv:2311.13373},
  year={2023}
}
```

## Acknowledgements
This work is supported by Exploratory Research Project (No.2022RC0AN02) of Zhejiang Lab.
