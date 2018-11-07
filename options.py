""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataroot',
                                 default='datasets/train/',
                                 help='path to dataset')
        self.parser.add_argument('--batchsize',
                                 type=int,
                                 default=64,
                                 help='input batch size')
        self.parser.add_argument('--workers',
                                 type=int,
                                 help='number of data loading workers',
                                 default=8)
        self.parser.add_argument('--imsize',
                                 type=int,
                                 default=128,
                                 help='input image size.')
        self.parser.add_argument('--nc',
                                 type=int,
                                 default=3,
                                 help='input image channels')
        self.parser.add_argument('--device',
                                 type=str,
                                 default='gpu',
                                 help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids',
                                 type=str,
                                 default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use "" for CPU')
        self.parser.add_argument('--ngpu',
                                 type=int,
                                 default=1,
                                 help='number of GPUs to use')
        self.parser.add_argument('--model',
                                 type=str,
                                 default='in4',
                                 help='chooses which model to use.')
        self.parser.add_argument('--outf',
                                 type=str,
                                 default='./output',
                                 help='folder to output options')
        self.parser.add_argument('--log_dir',
                                 type=str,
                                 default='./log',
                                 help='folder to log file and chekpoint saving')
        self.parser.add_argument('--eval_type',
                                 type=str,
                                 default='test',
                                 help='evaluate test or validate set')
        self.parser.add_argument('--image_path',
                                 type=str,
                                 default='',
                                 help='predict image file path')
        self.parser.add_argument('--model_file_name',
                                 type=str,
                                 default='/',
                                 help='file of model to predict')
        self.parser.add_argument('--n_class',
                                 type=int,
                                 default=2,
                                 help='number of class to predict')


        ##
        # Train
        self.parser.add_argument('--print_freq',
                                 type=int,
                                 default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--epochs',
                                 type=int,
                                 default=10,
                                 help='number of epochs to train for')
        self.parser.add_argument('--beta1',
                                 type=float,
                                 default=0.5,
                                 help='momentum term of adam')
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=0.0002,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--meta_graph_name',
                                 type=str,
                                 default='',
                                 help='load graph for image prediction')
        self.parser.add_argument('--is_train',
                                 type=bool,
                                 default=False,
                                 help='set True/False by train or test')

        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.outf, self.opt.model, 'train')
        # test_dir = os.path.join(self.opt.outf, self.opt.model, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        # if not os.path.isdir(test_dir):
        #     os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
