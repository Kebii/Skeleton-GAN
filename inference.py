import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.base_net import ST_Gen

def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='Skeleton_GAN')
    parser.add_argument(
        '--config',
        default='./config/cfg.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--weight',
        default='./saved/sk_wgan_50T/weights/sk_wgan_gen-699.pt',
        help='xxx.pt weight for generator')   
    parser.add_argument(
        '--test-save-path',
        default='./saved/sk_wgan/fake_sk_test',
        help='fake skeleton saved name')
    parser.add_argument(
        '--phase',
        default='test',
        help='must be train or test')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--T',
        type=int,
        default=300,
        help='generate sequence length')
    parser.add_argument(
        '--B',
        type=int,
        default=10,
        help='generate sequence number')
    parser.add_argument(
        '--gen-model-args',
        type=dict,
        default=dict(),
        help='the arguments of generator')
    
    return parser


def getmodel(weight_path, arg):
    model = ST_Gen(**arg.gen_model_args).cuda(arg.device[0])
    model = nn.DataParallel(model, device_ids=arg.device)

    print("load weight from: "+weight_path)
    weights = torch.load(weight_path)
    model.load_state_dict(weights)
    model.eval()
    return model

def generate_skeleton(model, arg):
    input_X = Variable(torch.randn(arg.B, 3, arg.T, arg.gen_model_args["joint_num"]).float().cuda(arg.device[0]), requires_grad=False)
    fake_X = model(input_X).cpu().detach().numpy()
    path = arg.test_save_path
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"/fake_skeleton", fake_X)
    print(path+"/fake_skeleton.npy has been saved!")

def main(arg):
    generator = getmodel(arg.weight, arg)
    generate_skeleton(generator, arg)



if __name__ == '__main__':
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    
    main(arg)
