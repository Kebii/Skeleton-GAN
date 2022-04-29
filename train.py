import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import time
import datetime
import random
import yaml
import argparse
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.autograd import grad as torch_grad

from feeder.feeder_ntu import Feeder
from model.base_net import ST_Gen, ST_Dis

def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='Skeleton_GAN')
    parser.add_argument(
        '--config',
        default='./config/cfg.yaml',
        help='path to the configuration file')   
    parser.add_argument(
        '--model-save-name',
        default='./saved/sk_wgan/sk_wgan',
        help='model saved name')
    parser.add_argument(
        '--val-save-name',
        default='./saved/sk_wgan/fake_sk',
        help='fake skeleton saved name')
    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.1,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--gen-model-args',
        type=dict,
        default=dict(),
        help='the arguments of generator')
    parser.add_argument(
        '--dis-model-args',
        type=dict,
        default=dict(),
        help='the arguments of discriminator')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='training batch size')
    
    
    return parser

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(generator, discriminator, data_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d, loss_func_g, loss_func_d, epoch, loop_d, logger, arg):
    pbar = tqdm(total=len(data_loader), ncols=140)
    epoch_loss_g = AverageMeter()
    epoch_loss_d = AverageMeter()
    epoch_time = AverageMeter()

    for batch_idx, (data, action_label, index) in  enumerate(data_loader):
        real_data = Variable(data.float().cuda(arg.device[0]), requires_grad=False)
        pbar.set_description("Train Epoch %i  Step %i" %(epoch, batch_idx))
        start_time = time.time()

        # train generator
        generator.train()
        discriminator.eval()
        optimizer_g.zero_grad()

        input_X = Variable(torch.randn(data.shape).float().cuda(arg.device[0]), requires_grad=False)
        fake_X = generator(input_X)
        pred_fake_X = discriminator(fake_X)
        loss_g = loss_func_g(pred_fake_X) * 1e9
        loss_g.backward()

        optimizer_g.step()
        scheduler_g.step()

        # train discriminator
        generator.eval()
        discriminator.train()
        for i in range(loop_d):

            optimizer_d.zero_grad()

            # fake_data = fake_X.clone().detach()
            input_dX = Variable(torch.randn(data.shape).float().cuda(arg.device[0]), requires_grad=False)
            fake_data = generator(input_dX).detach()
            pred_real_X = discriminator(real_data)
            pred_fake_X = discriminator(fake_data)
            loss_d = loss_func_d(pred_real_X, pred_fake_X) * 1e9
            # loss_gp = gradient_penalty(discriminator, real_data, fake_data)

            # loss_d = loss_d + 10*loss_gp
            loss_d.backward()
            optimizer_d.step()
        scheduler_d.step()

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        epoch_loss_g.update(float(loss_g.item()))
        epoch_loss_d.update(float(loss_d.item()))

        pbar.set_postfix(loss_g=float(loss_g.item()), loss_d=float(loss_d.item()), time=end_time-start_time)
        pbar.update(1)
    pbar.close()

    logger.add_scalar('train_loss_g', epoch_loss_g.avg, epoch)
    logger.add_scalar('train_loss_d', epoch_loss_d.avg, epoch)

    return epoch_loss_g, epoch_loss_d, epoch_time

def val(generator, b, c, t, v, path, epoch_i, arg):
    generator.eval()
    input_X = Variable(torch.randn(b,c,t,v).float().cuda(arg.device[0]), requires_grad=False)
    fake_X = generator(input_X).cpu().detach().numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"/fake_skeleton-"+str(epoch_i), fake_X)




def get_loss_fun_g():
    def loss_fun(x):
        return -torch.mean(x)
    return loss_fun

def get_loss_fun_d():
    def loss_fun(xr, xf):
        return torch.mean(xf) - torch.mean(xr)
    return loss_fun

def gradient_penalty(discriminator, real_data, generated_data):
    batch_size = real_data.size()[0]
    device_gpu = generated_data.device
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).cuda(device_gpu)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda(device_gpu)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(device_gpu),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return  ((gradients_norm - 1) ** 2).mean()



def main(arg):
    data_feeder = Feeder(**arg.train_feeder_args)
    generator_net = ST_Gen(**arg.gen_model_args).cuda(arg.device[0])
    discriminator_net = ST_Dis(**arg.dis_model_args).cuda(arg.device[0])
    generator_net = nn.DataParallel(generator_net, device_ids=arg.device)
    discriminator_net = nn.DataParallel(discriminator_net, device_ids=arg.device)

    data_loader = torch.utils.data.DataLoader(dataset=data_feeder, batch_size=arg.batch_size, num_workers=8, shuffle=True)
    optimizer_gen = optim.Adam(generator_net.parameters(), lr=arg.base_lr, weight_decay=arg.weight_decay)
    optimizer_dis = optim.Adam(discriminator_net.parameters(), lr=arg.base_lr, weight_decay=arg.weight_decay)

    scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(optimizer_gen, milestones=arg.step, gamma=0.1, last_epoch=-1)
    scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(optimizer_dis, milestones=arg.step, gamma=0.1, last_epoch=-1)

    loss_func_gen = get_loss_fun_g()
    loss_func_dis = get_loss_fun_d()

    train_writer = SummaryWriter(os.path.join(arg.model_save_name, 'train_log'), 'train')

    for i in range(arg.epoch):
        epoch_loss_g, epoch_loss_d, epoch_time = train(generator_net, discriminator_net, data_loader,
                                                        optimizer_gen, optimizer_dis, scheduler_gen, scheduler_dis,
                                                        loss_func_gen, loss_func_dis, i, 3, train_writer, arg)
        print('epoch:'+str(i)+"  gen loss:"+str(epoch_loss_g.avg) +"  dis loss:"+str(epoch_loss_d.avg))

        if (i+1) % 10 == 0:
            state_dict_gen = generator_net.state_dict()
            state_dict_dis = discriminator_net.state_dict()

            weights_gen = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_gen.items()])
            torch.save(weights_gen, arg.model_save_name + '_gen-' + str(i) + '.pt')
            print(arg.model_save_name + '_gen-' + str(i) + '.pt has been saved!')

            weights_dis = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_dis.items()])
            torch.save(weights_dis, arg.model_save_name + '_dis-' + str(i) + '.pt')
            print(arg.model_save_name + '_dis-' + str(i) + '.pt has been saved!')
        if (i+1) % 10 == 0:
            val(generator_net, 64, 3, 50, 25, arg.val_save_name, i, arg)
            print(arg.val_save_name + '/fake_skeleton-' + str(i) + '.npy has been saved!')



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