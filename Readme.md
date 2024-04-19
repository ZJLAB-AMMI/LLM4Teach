# [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373)

## Abstract 
Recent studies have uncovered the potential of Large Language Models (LLMs) in addressing complex sequential decision-making tasks through the provision of high-level instructions. However, LLM-based agents lack specialization in tackling specific target problems, particularly in real-time dynamic environments. Additionally, deploying an LLM-based agent in practical scenarios can be both costly and time-consuming. On the other hand, reinforcement learning (RL) approaches train agents that specialize in the target task but often suffer from low sampling efficiency and high exploration costs. In this paper, we introduce a novel framework that addresses these challenges by training a smaller, specialized student RL agent using instructions from an LLM-based teacher agent. By incorporating the guidance from the teacher agent, the student agent can distill the prior knowledge of the LLM into its own model. Consequently, the student agent can be trained with significantly less data. Moreover, through further training with environment feedback, the student agent surpasses the capabilities of its teacher for completing the target task. We conducted experiments on challenging MiniGrid and Habitat environments, specifically designed for embodied AI research, to evaluate the effectiveness of our framework. The results clearly demonstrate that our approach achieves superior performance compared to strong baseline methods. Our code is available at https://github.com/ZJLAB-AMMI/LLM4Teach.

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373).


## Running experiments
### Setup the LLMs

1. For ChatGLM models, please use your own api_key and run the following code to launch the API
```bash
python3 -m utils.chatglm_api --host <API_host> --port <API_port>
```

2. For Vicuna models, please follow the instruction from [FastChat](https://github.com/lm-sys/FastChat) to install Vicuna model on local sever. Here are the commands to launch the API in terminal: 

```bash
python3 -m fastchat.serve.controller --host localhost --port <controller_port>        ### Launch the controller
python3 -m fastchat.serve.model_worker --model-name '<model_name>' --model-path <Vicuna_path> --controller http://localhost:<controller_port> --port <model_port> --worker_address http://localhost:<model_port>        ### Launch the model worker
python3 -m fastchat.serve.api --host <API_host> --port <API_port>        ### Launch the API
```


### Train and evaluate the models
Any algorithm can be run from the main.py entry point.

To train on a SimpleDoorKey environment,

```bash
python main.py train --task SimpleDoorKey --savedir train
```

<!--to train with given query result from LLM as teacher,

```bash
python main.py train --task SimpleDoorKey --savedir train --offline_planner
```-->

To evaluate the trained model,

```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval
```

To evaluate the LLM-based teacher baseline,
```bash
python main.py eval --task SimpleDoorKey --loaddir train --savedir eval --eval_teacher
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
@inproceedings{zhou2024large,
  title={Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents},
  author={Zhou, Zihao and Hu, Bin and Zhao, Chenyang and Zhang, Pu and Liu, Bin},
  booktitle={The 33rd International Joint Conference on Artificial Intelligence (IJCAI 2024)},
  year={2024}
}
```

## Acknowledgements
This work is supported by Exploratory Research Project (No.2022RC0AN02) of Zhejiang Lab.
