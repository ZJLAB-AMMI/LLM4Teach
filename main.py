import argparse
import os,json, sys
import numpy as np
# single gpu

os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

import torch
import utils
from Game import Game
    
def train(args):
    for i in args.seed_list:
        args.savedir = args.savedir + "-" + str(i)
        args.seed = i
        game = Game(args)
        game.train()
        
def evaluate(args):
    assert args.loaddir
    print("env name: %s for %s" %(args.task, args.loaddir))
    args.seed = args.seed_list[0]
    game = Game(args, training=False)
    eval_returns = []
    eval_lens = []
    eval_success = []
    
    if len(args.env_seed_list) == 0:
        env_seed_list = [None] * args.num_eval
    elif len(args.env_seed_list) == 1:
        env_seed_list = [args.env_seed_list[0] + i for i in range(args.num_eval)]
    else:
        env_seed_list = args.env_seed_list
        
    for i in env_seed_list:
        eval_outputs = game.evaluate(seed = i, teacher_policy = args.eval_teacher)
        eval_returns.append(eval_outputs[0])
        eval_lens.append(eval_outputs[1])
        eval_success.append(eval_outputs[2])

    print("Mean return:", np.mean(eval_returns))
    print("Mean length:", np.mean(eval_lens))
    print("Success rate:", np.mean(eval_success))


if __name__ == "__main__":
    utils.print_logo(subtitle="Maintained by Research Center for Applied Mathematics and Machine Intelligence, Zhejiang Lab")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SimpleDoorKey", help="SimpleDoorKey, KeyInBox, RandomBoxKey, ColoredDoorKey, DynamicDoorKey") 
    
    # parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--env_seed_list", type=int, nargs="*", default=[0], help="Seeds for evaluation environments")
    parser.add_argument("--seed_list", type=int, nargs="*", default=[0], help="Seeds for Numpy, Torch and LLM")
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy", type=str, default='ppo')
    parser.add_argument("--n_itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--recurrent", default=False, action='store_true')
    
    parser.add_argument("--logdir", type=str, default="log") # Where to log diagnostics to
    parser.add_argument("--loaddir", type=str, default=None)
    parser.add_argument("--loadmodel", type=str, default="acmodel")
    parser.add_argument("--savedir", type=str, required=True, help="path to folder containing policy and run details")
    
    parser.add_argument("--offline_planner", default=False, action='store_true')
    parser.add_argument("--soft_planner", default=False, action='store_true')
    parser.add_argument("--eval_teacher", default=False, action='store_true')
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    
    if sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        evaluate(args)
    elif sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        train(args)
    else:
        print("Invalid option '{}'".format(sys.argv[1]))