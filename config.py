import argparse
import torch
import numpy as np
import random


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of DeepLearning')
        # parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        # parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=0.001, type=float, help='initial weight decay')
        parser.add_argument('--epoch', default=50, type=int, help='number of epochs for training')
        parser.add_argument('--MaxEpoch', default=5, type=int, help='earlystop')
        parser.add_argument('--batch_size', default=8, type=int, help='number of batch_size')
        parser.add_argument('--model', default='Unet', type=str, help='Unet, EGEUnet')
        parser.add_argument('--optimizer', default='Adam', type=str, help='Adam, SGD')
        parser.add_argument('--momentum', default=0.9, type=float, help='Adam, SGD')

        parser.add_argument('--size', default=256, type=int, help='size of input image')
        parser.add_argument('--dropout', default=0.1, type=float, help='number of batch_size')
        parser.add_argument('--seed', type=int, default=2024, help='random seed to set')
        parser.add_argument('--num_workers', type=int, default=4, help='')
        parser.add_argument('--log_path', type=str, default='./results')
        parser.add_argument('--dataset', type=str, default='ISIC2018', help='ISIC2018, XiangYa')
        args = parser.parse_args()
        # args.time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

    def initialize(self):
        self.set_seed(self.args.seed)
        return self.args

    def set_seed(self, seed=1000):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)