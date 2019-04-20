'''trainning/testing options
'''

import sys
import argparse
import json
import os
import re

import torch

from utils import utils, keys
from options.options_utils import str2bool, print_opt, confirm_opt

def set_general_arguments(parser):
    general_arg = parser.add_argument_group('General')
    general_arg.add_argument('--confirm', type=str2bool, default=True, help='promote confirmation for user')

def set_data_arguments(parser):
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--shuffle_data', type=str2bool, default=True, help='use sequence dataset or shuffled dataset')
    data_arg.add_argument('--load_rgb', type=str2bool, default=True, help='rgb guided')
    data_arg.add_argument('--rgb2gray', type=str2bool, default=False, help='convert rgb to grayscale')
    data_arg.add_argument('--invert_depth', type=str2bool, default=False, help='convert depth to disparity')
    data_arg.add_argument('--norm_factor', type=float, default=256.0, help='normalize the depth image')
    data_arg.add_argument('--workers', type=int, default=0, help='num of workers to fetch data')
    data_arg.add_argument('--data_cut', choices=[100, 10, 1], type=int, default=1, help='1/100 data, 1/10 data or all data')

def set_voronoi_arguments(parser):
    vor_arg = parser.add_argument_group('Voronoi')
    vor_arg.add_argument('--vor_radius', type=float, default=10.0, help='radius of the circle/cone to render voronoi diagram')

def set_options(training: bool):
    '''
    This function will return an option object that
    contains all the training/testing options.

    Arguments:
        training {bool} -- [indicating training]
    '''
    try:
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'global_config.json'), 'r') as f:
            global_config = json.load(f)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        exit(1)

    parser = argparse.ArgumentParser()
    set_general_arguments(parser)
    set_data_arguments(parser)
    set_voronoi_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=global_config['out'], help='out directory')
    parser.add_argument('--tb_dir', type=str, default=global_config['runs'], help='tensorboard runs directory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--learning_type', choices=[keys.UNGUIDED_COMPLETION, keys.GUIDED_COMPLETION], type=str, default='guided', help='un/guided depth completion')
    parser.add_argument('--residual_learning', type=str2bool, default=True, help='using NN filling and then learning the residual')
    parser.add_argument('--max_iter', type=int, default=10000, help='total training iterations')
    parser.add_argument('--valid_iter', type=int, default=300, help='iterval of validation')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume training with same model name')
    parser.add_argument('--load_weights', type=str, default=None, help='load weights from existing models')
    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)
    opt.use_cuda = True
    if opt.use_cuda:
        assert torch.cuda.is_available()

    opt.data_path = global_config['data_path']
    opt.gt_path = global_config['gt_path']
    opt.rgb_path = global_config['rgb_path']
    opt.name = 'test_train_sym'
    opt.out = os.path.join(opt.out_dir, opt.name)
    opt.tb_out = os.path.join(opt.tb_dir, opt.name)

    if opt.confirm:
        confirm_opt(opt)
    else:
        print_opt(opt)

    return opt
