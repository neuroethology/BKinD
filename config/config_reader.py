import argparse

import yaml


def create_parser():
    parser = argparse.ArgumentParser(description='Training Keypoint Discovery')

    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', default='CalMS21', type=str,
                        help='dataset name')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--val-schedule', default=5, type=int,
                        help='evaluate on validation set every N epochs')
    parser.add_argument('--visualize', default=True,
                        help='visualize discovered keypoints, reconstruction, and heatmaps')

    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--nkpts', default=10, type=int, metavar='N',
                        help='number of keypoints')
    parser.add_argument('--schedule', default=30, type=int, metavar='N',
                        help='learning rate scheduler')
    parser.add_argument('--image-size', default=128, type=int, metavar='N',
                        help='input image size')
    parser.add_argument('--perc-weight', default=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0], type=list,
                        help='weights for the perceptual loss')
    parser.add_argument('--bounding-box', default=False, 
                        help='use cropped bounding box for training')
    parser.add_argument('--curriculum', default=4, type=int,
                        help='apply rotation equivariance loss after N epochs')

    parser.add_argument('--config', default='config/CalMS21.yaml',
                        type=argparse.FileType(mode='r'))

    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config:
        data = yaml.safe_load(args.config)
        delattr(args, 'config')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    return args
