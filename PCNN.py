import torch
from flownet import FlowNet
from vggf import VGGF,multiClassHingeLoss

import scipy.io as sio

import os
import sys
import copy
import torch.multiprocessing

import string
import random
import argparse
import time
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from dataset import PCNN_VideoDataset

from tqdm import tqdm
import math

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

parser = argparse.ArgumentParser()
parser.add_argument('--n_worker', default=0, type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--run_id', default='', type=str, metavar='run_id')
parser.add_argument('--dump_dir', default='./logs/train_feature', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--step', default=-1, type=int)

parser.add_argument('--n_seg', default=1, type=int)
parser.add_argument('--seq_len', default=1, type=int)

parser.add_argument('--max_step', default=3570, type=int)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lr_decay_mode', default='step', type=str)
parser.add_argument('--lr_decay_step', nargs='+', default=-1, type=int)

parser.add_argument('--lr_decay_gamma', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--grad_clip', default=100, type=float)

parser.add_argument('--acc_grad', default=0, type=int)

parser.add_argument('--save_every_step', default=700, type=float)
parser.add_argument('--update_every_step', default=1, type=int)

parser.add_argument('--evaluate', action='store_true')

parser.add_argument('--info_basedir', type=str, default='/data6/SRIP18-7/codes/bench-video/dataset/hmdb51/info_data')
parser.add_argument('--feature_name1', type=str, default = 'rgb')
parser.add_argument('--feature_name2', type=str, default = 'flow-maps')
parser.add_argument('--feature_name3', type=str, default = 'pose')
parser.add_argument('--feature_dim', type=int)
parser.add_argument('--split', default='0', type=str)

args = parser.parse_args()
if args.run_id == '':
    args.run_id = id_generator()
    print('run_id: {}'.format(args.run_id))
if args.update_every_step == -1:
    args.update_every_step = 128/args.batch_size

if args.max_step == -1:
    args.max_step = 3500 # doubled
else:
    args.max_step = args.max_step * args.update_every_step
if args.lr_decay_step == -1:
    args.lr_decay_step = [1500, 3000]
else:
    args.lr_decay_step = [i * args.update_every_step for i in args.lr_decay_step]

#args.save_every_step = max(args.max_step/100, 1) # doubled

args.gpu_id = args.gpu_id.split(',')
args.gpu_id = [int(r) for r in args.gpu_id]


torch.cuda.set_device(args.gpu_id[0])

from torchvision import transforms
from torch.utils.data import DataLoader

from torch import optim
from torch.autograd.variable import Variable
from saver import Saver, make_log_dirs

import json

from tensorboardX import SummaryWriter

def class_freqs(dataset):
    counts = torch.zeros(dataset.n_class)
    for data in list(dataset.infos.values())[0]:
        counts[data['label']] += 1
    freqs = counts / counts.sum()
    return freqs.numpy()


best_prec1 = 0
step = 0

def main():
    global step, best_prec1, svm_layer, rgbCNN, flowCNN, crit, optimizer, saver, writer
    global train_dataset, test_dataset, train_loader, test_loader, vocab

    print('prepare dataset...')
    (train_dataset, train_loader), (test_dataset, test_loader) = prepare_dataset()

    # prepare model
    
    svm_layer, rgbCNN, flowCNN, crit, optimizer = prepare_model()
    if args.resume != '':
        t_saver = Saver(model_dir=args.resume, max_to_keep=5)
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.evaluate:
            checkpoint = t_saver.load_best()
            # checkpoint = t_saver.load_latest()
        else:
            checkpoint = t_saver.load_latest()
        if checkpoint is not None:
            # torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            svm_layer.load_state_dict(checkpoint['model_state_dict'])
            if 'step' in checkpoint:
                step = checkpoint['step']
            else:
                step = 0
            if args.step != -1:
                step = args.step

            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.resume, checkpoint['step']))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(True)
        return
    
    print('prepare logger...')
    # writer, saver = prepare_logger()
    print('start training...')
    with torch.no_grad():
        feature_Matrix, labels = train()

    np.save('train_feature_list_hmdb.npy', np.array(feature_Matrix))
    np.save('train_labels_list_hmdb.npy', np.array(labels))
    print('done!')

def prepare_dataset():

    if args.n_seg == 1:
        seq_strides = list(range(5, 35)) + ['uniform', ]
    elif args.seq_len == 1 and args.n_seg == 3:
        seq_strides = (1, )
    else:
        raise RuntimeError('seq_len wrong!')

    
    train_dataset = PCNN_VideoDataset(info_basedir=args.info_basedir, phase='train', split=args.split, to_read=('video_name',
                                                                                                           'label',
                                                                                                           args.feature_name1,
                                                                                                           args.feature_name2,
                                                                                                           args.feature_name3),
                                 seq_len=1, n_seg=25, seq_strides=('uniform', ), shuffle=False)

    test_dataset = PCNN_VideoDataset(info_basedir=args.info_basedir, phase='test', split=args.split, to_read=('video_name',
                                                                                                         'label',
                                                                                                         args.feature_name1,
                                                                                                         args.feature_name2,
                                                                                                         args.feature_name3),
                                seq_len=1, n_seg=25, seq_strides=('uniform', ), shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                              pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                              pin_memory=False)


    return (train_dataset, train_loader), (test_dataset, test_loader)

def prepare_model():
    from collections import OrderedDict
    svm_layer = nn.Sequential(OrderedDict([('fc', nn.Linear(4096*4*5*2, train_dataset.n_class))]))
    rgbCNN = VGGF()
    flowCNN = FlowNet()

    crit = multiClassHingeLoss()

    svm_layer = svm_layer.cuda(args.gpu_id[0])
    rgbCNN = rgbCNN.cuda(args.gpu_id[0])
    flowCNN = flowCNN.cuda(args.gpu_id[0])

    optimizer = optim.SGD(params=svm_layer.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return svm_layer, rgbCNN, flowCNN, crit, optimizer

import subprocess, os, sys, datetime
def prepare_logger():
    model_dir, train_dir, log_dir = make_log_dirs(args.dump_dir, args.run_id)
    writer = SummaryWriter(log_dir)

    saver = Saver(model_dir=model_dir, max_to_keep=5)
    config_obj = dict(dataset_config=train_dataset.config, model_config=dict(model_name='softmax+ave'), train_config=vars(args))
    if not os.path.exists(os.path.join(log_dir, 'config.json')):
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
           json.dump(config_obj, f, indent=2)

    # Unbuffer output
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

    tee = subprocess.Popen(["tee", os.path.join(train_dir, datetime.datetime.now().strftime('output_%H_%M_%d_%m_%Y.log'))]
                           , stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    print(' '.join(sys.argv))

    return writer, saver


def check_for_nan(ps):
    for p in ps:
        if p is not None:
            if not np.isfinite(np.sum(p.data.cpu().numpy())):
                return True
    return False

def train():
    global step, best_prec1

    feature_Matrix = []
    label_all = []
    for iter_, (vid_names,labels,lhandrgbv,lhandflowv,rhandrgbv,rhandflowv,upperbodyrgbv,upperbodyflowv,fullbodyrgbv,fullbodyflowv,fullimagergbv,fullimageflowv) in enumerate(tqdm(train_loader)):
        # print(labels.min(), labels.max())

        rgbCNN.train()
        flowCNN.train()

        # print(labels.numpy())
        lhandrgbv = lhandrgbv.squeeze(0).cuda()
        lhandflowv = lhandflowv.squeeze(0).cuda()
        rhandrgbv = rhandrgbv.squeeze(0).cuda()
        rhandflowv = rhandflowv.squeeze(0).cuda()
        upperbodyrgbv = upperbodyrgbv.squeeze(0).cuda()
        upperbodyflowv = upperbodyflowv.squeeze(0).cuda()
        fullbodyrgbv = fullbodyrgbv.squeeze(0).cuda()
        fullbodyflowv = fullbodyflowv.squeeze(0).cuda()
        fullimagergbv = fullimagergbv.squeeze(0).cuda()
        fullimageflowv = fullimageflowv.squeeze(0).cuda()

        lhand_rbgfeature = rgbCNN(lhandrgbv).cpu().data
        rhand_rbgfeature = rgbCNN(rhandrgbv).cpu().data
        upperbody_rbgfeature = rgbCNN(upperbodyrgbv).cpu().data
        fullbody_rbgfeature = rgbCNN(fullbodyrgbv).cpu().data
        fullimage_rbgfeature = rgbCNN(fullimagergbv).cpu().data

        lhand_flowfeature = flowCNN(lhandflowv).cpu().data
        rhand_flowfeature = flowCNN(rhandflowv).cpu().data
        upperbody_flowfeature = flowCNN(upperbodyflowv).cpu().data
        fullbody_flowfeature = flowCNN(fullbodyflowv).cpu().data
        fullimage_flowfeature = flowCNN(fullimageflowv).cpu().data

        # print(lhand_rbgfeature.shape)

        lhand_rbgfeature_diff = []
        rhand_rbgfeature_diff = []
        upperbody_rbgfeature_diff = []
        fullbody_rbgfeature_diff = []
        fullimage_rbgfeature_diff = []

        lhand_flowfeature_diff = []
        rhand_flowfeature_diff = []
        upperbody_flowfeature_diff = []
        fullbody_flowfeature_diff = []
        fullimage_flowfeature_diff = []

        for i in range(lhandrgbv.shape[0] - 1):
            lhand_rbgfeature_diff.append(lhand_rbgfeature[i+1,:] - lhand_rbgfeature[i,:])
            rhand_rbgfeature_diff.append(rhand_rbgfeature[i+1,:] - rhand_rbgfeature[i,:])
            upperbody_rbgfeature_diff.append(upperbody_rbgfeature[i+1,:] - upperbody_rbgfeature[i,:])
            fullbody_rbgfeature_diff.append(fullbody_rbgfeature[i+1,:] - fullbody_rbgfeature[i,:])
            fullimage_rbgfeature_diff.append(fullimage_rbgfeature[i+1,:] - fullimage_rbgfeature[i,:])

            lhand_flowfeature_diff.append(lhand_flowfeature[i+1,:] - lhand_flowfeature[i,:])
            rhand_flowfeature_diff.append(rhand_flowfeature[i+1,:] - rhand_flowfeature[i,:])
            upperbody_flowfeature_diff.append(upperbody_flowfeature[i+1,:] - upperbody_flowfeature[i,:])
            fullbody_flowfeature_diff.append(fullbody_flowfeature[i+1,:] - fullbody_flowfeature[i,:])
            fullimage_flowfeature_diff.append(fullimage_flowfeature[i+1,:] - fullimage_flowfeature[i,:])

        lhand_rbgfeature_diff = torch.stack(lhand_rbgfeature_diff)
        rhand_rbgfeature_diff = torch.stack(rhand_rbgfeature_diff)
        upperbody_rbgfeature_diff = torch.stack(upperbody_rbgfeature_diff)
        fullbody_rbgfeature_diff = torch.stack(fullbody_rbgfeature_diff)
        fullimage_rbgfeature_diff = torch.stack(fullimage_rbgfeature_diff)
        # print(lhand_rbgfeature_diff.shape)

        lhand_flowfeature_diff = torch.stack(lhand_flowfeature_diff)
        rhand_flowfeature_diff = torch.stack(rhand_flowfeature_diff)
        upperbody_flowfeature_diff = torch.stack(upperbody_flowfeature_diff)
        fullbody_flowfeature_diff = torch.stack(fullbody_flowfeature_diff)
        fullimage_flowfeature_diff = torch.stack(fullimage_flowfeature_diff)

        lhand_rbgfeature_statM = torch.max(lhand_rbgfeature,0)[0]
        lhand_rbgfeature_statm = torch.min(lhand_rbgfeature,0)[0]
        lhand_rbgfeature_dynM = torch.max(lhand_rbgfeature_diff,0)[0]
        lhand_rbgfeature_dynm = torch.min(lhand_rbgfeature_diff,0)[0]
        lhand_flowfeature_statM = torch.max(lhand_flowfeature,0)[0]
        lhand_flowfeature_statm = torch.min(lhand_flowfeature,0)[0]
        lhand_flowfeature_dynM = torch.max(lhand_flowfeature_diff,0)[0]
        lhand_flowfeature_dynm = torch.min(lhand_flowfeature_diff,0)[0]
        # print(lhand_rbgfeature_statM.shape)

        rhand_rbgfeature_statM = torch.max(rhand_rbgfeature,0)[0]
        rhand_rbgfeature_statm = torch.min(rhand_rbgfeature,0)[0]
        rhand_rbgfeature_dynM = torch.max(rhand_rbgfeature_diff,0)[0]
        rhand_rbgfeature_dynm = torch.min(rhand_rbgfeature_diff,0)[0]
        rhand_flowfeature_statM = torch.max(rhand_flowfeature,0)[0]
        rhand_flowfeature_statm = torch.min(rhand_flowfeature,0)[0]
        rhand_flowfeature_dynM = torch.max(rhand_flowfeature_diff,0)[0]
        rhand_flowfeature_dynm = torch.min(rhand_flowfeature_diff,0)[0]

        upperbody_rbgfeature_statM = torch.max(upperbody_rbgfeature,0)[0]
        upperbody_rbgfeature_statm = torch.min(upperbody_rbgfeature,0)[0]
        upperbody_rbgfeature_dynM = torch.max(upperbody_rbgfeature_diff,0)[0]
        upperbody_rbgfeature_dynm = torch.min(upperbody_rbgfeature_diff,0)[0]
        upperbody_flowfeature_statM = torch.max(upperbody_flowfeature,0)[0]
        upperbody_flowfeature_statm = torch.min(upperbody_flowfeature,0)[0]
        upperbody_flowfeature_dynM = torch.max(upperbody_flowfeature_diff,0)[0]
        upperbody_flowfeature_dynm = torch.min(upperbody_flowfeature_diff,0)[0]

        fullbody_rbgfeature_statM = torch.max(fullbody_rbgfeature,0)[0]
        fullbody_rbgfeature_statm = torch.min(fullbody_rbgfeature,0)[0]
        fullbody_rbgfeature_dynM = torch.max(fullbody_rbgfeature_diff,0)[0]
        fullbody_rbgfeature_dynm = torch.min(fullbody_rbgfeature_diff,0)[0]
        fullbody_flowfeature_statM = torch.max(fullbody_flowfeature,0)[0]
        fullbody_flowfeature_statm = torch.min(fullbody_flowfeature,0)[0]
        fullbody_flowfeature_dynM = torch.max(fullbody_flowfeature_diff,0)[0]
        fullbody_flowfeature_dynm = torch.min(fullbody_flowfeature_diff,0)[0]

        fullimage_rbgfeature_statM = torch.max(fullimage_rbgfeature,0)[0]
        fullimage_rbgfeature_statm = torch.min(fullimage_rbgfeature,0)[0]
        fullimage_rbgfeature_dynM = torch.max(fullimage_rbgfeature_diff,0)[0]
        fullimage_rbgfeature_dynm = torch.min(fullimage_rbgfeature_diff,0)[0]
        fullimage_flowfeature_statM = torch.max(fullimage_flowfeature,0)[0]
        fullimage_flowfeature_statm = torch.min(fullimage_flowfeature,0)[0]
        fullimage_flowfeature_dynM = torch.max(fullimage_flowfeature_diff,0)[0]
        fullimage_flowfeature_dynm = torch.min(fullimage_flowfeature_diff,0)[0]

        feature_final = torch.cat([lhand_rbgfeature_statM,lhand_rbgfeature_statm,lhand_rbgfeature_dynM,lhand_rbgfeature_dynm,\
        rhand_rbgfeature_statM,rhand_rbgfeature_statm,rhand_rbgfeature_dynM,rhand_rbgfeature_dynm,\
        upperbody_rbgfeature_statM,upperbody_rbgfeature_statm,upperbody_rbgfeature_dynM,upperbody_rbgfeature_dynm,\
        fullbody_rbgfeature_statM,fullbody_rbgfeature_statm,fullbody_rbgfeature_dynM,fullbody_rbgfeature_dynm,\
        fullimage_rbgfeature_statM,fullimage_rbgfeature_statm,fullimage_rbgfeature_dynM,fullimage_rbgfeature_dynm,\
        lhand_flowfeature_statM,lhand_flowfeature_statm,lhand_flowfeature_dynM,lhand_flowfeature_dynm,\
        rhand_flowfeature_statM,rhand_flowfeature_statm,rhand_flowfeature_dynM,rhand_flowfeature_dynm,\
        upperbody_flowfeature_statM,upperbody_flowfeature_statm,upperbody_flowfeature_dynM,upperbody_flowfeature_dynm,\
        fullbody_flowfeature_statM,fullbody_flowfeature_statm,fullbody_flowfeature_dynM,fullbody_flowfeature_dynm,\
        fullimage_flowfeature_statM,fullimage_flowfeature_statm,fullimage_flowfeature_dynM,fullimage_flowfeature_dynm])

        feature_Matrix.append(feature_final.numpy())
        label_all.append(labels.numpy()[0])

    mean_feature_vec = sum(feature_Matrix)/len(feature_Matrix)
    feature_Matrix_final = [feature_vec - mean_feature_vec for feature_vec in feature_Matrix]

    return feature_Matrix_final, label_all

def poly_adjust_learning_rate(optimizer, lr0, step, n_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (1.0 - step*1.0/n_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def step_adjust_learning_rate(optimizer, lr0, step, step_size, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if len(step_size) == 0:
        lr = lr0 * (gamma ** (step // step_size))
    else:
        lr = lr0 * gamma ** (sum([step > i for i in step_size]))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def my_save_checkpoint(saver, model, info_dict, is_best, step):
    saver.save(model=model, info_dict=info_dict, is_best=is_best, step=step)

if __name__ == '__main__':
    main()
