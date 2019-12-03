import argparse
import shutil
import os

def parse_args():
    descript = 'Pytorch Implementation of Background Suppression Network (BaS-Net)'
    parser = argparse.ArgumentParser(description=descript)

    parser.add_argument('--data_path', type=str, default='./dataset/THUMOS14')
    parser.add_argument('--model_path', type=str, default='./models/BaSnet')
    parser.add_argument('--output_path', type=str, default='./outputs/BaSnet')
    parser.add_argument('--log_path', type=str, default='./logs/BaSnet')
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'])
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--class_th', type=float, default=0.25)
    parser.add_argument('--lr', type=str, default='[0.0001]*1500', help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--debug', action='store_true')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args