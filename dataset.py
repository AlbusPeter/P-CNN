import os
import torch
import torchvision

from torch.utils.data import Dataset
from config_wrapper import ConfigWrapper
from utils import default_get_frame_inds, video_loader_transfomer, h5_loader, img_loader, jpg_loader, flow_loader
import json
import numpy as np
import math
import cv2
import collections
import copy
import random
from PIL import Image
LOADER_DICT = dict(video=video_loader_transfomer, feature=h5_loader, img=img_loader, jpg=jpg_loader, jpg_flow=flow_loader)

class PCNN_VideoDataset(Dataset, ConfigWrapper):
    def __init__(self, info_basedir, phase='', split='', to_read=(), seq_len=1, n_seg=3, seq_strides=('uniform', ),
                 aug_video=True, transformer=collections.defaultdict(lambda: None), run_n_sample=0, shuffle=True,
                 classes=None):

        # config handler
        attrs = locals()
        ConfigWrapper.__init__(self, attrs)
        # info loader
        with open(os.path.join(info_basedir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.n_class = self.dataset_info['n_class']
        infos = dict()
        t_n_sample = None
        for mod_name in self.dataset_info['modality'].keys():
            mod = self.dataset_info['modality'][mod_name]
            if os.path.exists(os.path.join(info_basedir, '{}_{}_{}.json'.format(phase, mod_name, split))):
                with open(os.path.join(info_basedir, '{}_{}_{}.json'.format(phase, mod_name, split)), 'r') as f:
                    infos[mod_name] = json.load(f)
                    # keep classes
                    if classes:
                        infos[mod_name] = [x for x in infos[mod_name] if x['label'] in classes]
                    
                    if t_n_sample is None:
                        t_n_sample = len(infos[mod_name])
                    else:
                        if len(infos[mod_name]) != t_n_sample:
                            RuntimeError('sample number wrong for modality {}'.format(mod_name))
            # else:
            #     print('warning, missing info for modal:{}'.format(mod_name))
            # register mode loader
            def get_factory(mod_name):
                def getter(item, context):
                    return self._get_mode(mod_name, item, context)
                return getter
            self.__setattr__('get_{}'.format(mod_name), get_factory(mod_name))
        self.infos = infos

        # sample iterator
        if run_n_sample == 0:
            run_n_sample = t_n_sample
        self.run_n_sample = run_n_sample
        self.n_sample = t_n_sample
        
        n_epoch = int(math.ceil(self.run_n_sample * 1.0 / self.n_sample))

        n_sample_ind_iter = []
        for _ in range(n_epoch):
            if shuffle:
                iter_epoch = np.random.permutation(self.n_sample).tolist()
            else:
                iter_epoch = range(self.n_sample)
            n_sample_ind_iter = n_sample_ind_iter + list(iter_epoch)
        n_sample_ind_iter = n_sample_ind_iter[:self.run_n_sample]
        self.n_sample_ind_iter = n_sample_ind_iter

        self.righthand = 4
        self.lefthand = 7
        self.upperbody = [0,1,2,3,4,5,6,7,8,11,14,15,16,17]
        self.fullbody = [i for i in range(18)]
        self.box_length = 40

        #average images
        import scipy.io as sio
        vgg_model = sio.loadmat('/data6/SRIP18-7/codes/PCNN/pretrained_models/imagenet-vgg-f.mat')
        flownet_model = sio.loadmat('/data6/SRIP18-7/codes/PCNN/pretrained_models/flow_net.mat')
        self.rgb_avg = torch.from_numpy(vgg_model['normalization'][0][0][0]/255.0).float().permute(2,0,1)
        self.flow_avg = torch.from_numpy(flownet_model['normalization'][0][0][0]/255.0).float().permute(2,0,1)

    
    def get_box_and_fill(self,topleft,botright,im,flow_im):
        im_shape = im.shape #bacthsize*frame_num*3*h*w
        flow_im_shape = flow_im.shape

        box_h = botright[0]-topleft[0]+1
        box_w = botright[1]-topleft[1]+1
        box = torch.ones([3,box_h,box_w]) * 0.5
        box_flow = torch.ones([3,box_h,box_w]) * 0.5

        if botright[0] == self.box_length and botright[1] == self.box_length :
            return box, box_flow

        if topleft[0] > im_shape[-2] - 1 or topleft[1] > im_shape[-1] - 1 or botright[0] < 0 or botright[1] < 0 :
            return box, box_flow

        left_min = max(topleft[1],0)
        top_min = max(topleft[0],0)

        right_max = min(botright[1],im_shape[-1] - 1)
        bot_max = min(botright[0],im_shape[-2] - 1)

        new_w = right_max - left_min + 1
        new_h = bot_max - top_min + 1

        # print(im[:,top_min:bot_max+1,left_min:right_max+1])
        # print(flow_im[:,top_min:bot_max+1,left_min:right_max+1])

        box[:,top_min-topleft[0]:top_min-topleft[0]+new_h,left_min-topleft[1]:left_min-topleft[1]+new_w]=im[:,top_min:bot_max+1,left_min:right_max+1]
        box_flow[:,top_min-topleft[0]:top_min-topleft[0]+new_h,left_min-topleft[1]:left_min-topleft[1]+new_w]=flow_im[:,top_min:bot_max+1,left_min:right_max+1]

        return box, box_flow

    def get_random_frame_inds(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = list(self.infos.values())[0][item]
        n_frame = info['end_frame'] - info['start_frame']

        frame_inds = default_get_frame_inds(n_frame=n_frame, n_seg=self.n_seg, seq_strides=self.seq_strides,
                                            seq_len=self.seq_len)
        # frame_inds += info['start_frame']

        context['frame_inds'] = frame_inds

    def get_label(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = list(self.infos.values())[0][item]
        label = info['label']
        context['label'] = label
        return label

    def get_vid_name(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = list(self.infos.values())[0][item]
        context['vid_name'] = info['vid_name']
        return context['vid_name']

    def _get_mode(self, mode, item, context):
        item = self.n_sample_ind_iter[item]
        vid_name = list(self.infos.values())[0][item]['vid_name']

        mode_path = os.path.join(self.dataset_info['modality'][mode]['mode_basedir'], vid_name+self.dataset_info['modality'][mode]['mode_ext'])
        t = None
        if mode in self.transformer:
            t = self.transformer[mode]

        if 'config' not in self.dataset_info['modality'][mode]:
            self.dataset_info['modality'][mode]['config'] = dict()
        
        # try:
        context[mode] = LOADER_DICT[self.dataset_info['modality'][mode]['mode_format']](mode_path,
                                                                                        context['frame_inds'],
                                                                                        t,
                                                                                        self.dataset_info['modality'][
                                                                                            mode]['config']
                                                                                        )
        # except:
            # print(mode_path)
        return context[mode]

    def get_video_name(self, item, context):
        item = self.n_sample_ind_iter[item]
        vid_name = list(self.infos.values())[0][item]['vid_name']
        context['vid_name'] = vid_name
        return vid_name

    def __getitem__(self, item):
        context = dict()
        self.get_random_frame_inds(item, context)
        output = []
        for key in self.to_read:
            method = getattr(self, 'get_{}'.format(key))
            val = method(item, context)
            output.append(val)
        
        lhandrgbv = []
        lhandflowv = []
        rhandrgbv = []
        rhandflowv = []
        upperbodyrgbv = []
        upperbodyflowv = []
        fullbodyrgbv = []
        fullbodyflowv = []
        fullimagergbv = []
        fullimageflowv = []

        RGB_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize([224,224]),
            torchvision.transforms.ToTensor()
        ]

        Flow_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop([227,227]),
            torchvision.transforms.ToTensor()
        ]

        transformer_RGB = torchvision.transforms.Compose(RGB_transforms)
        transformer_Flow = torchvision.transforms.Compose(Flow_transforms)

        # print(output[2].shape[0])
        # print(output[3].shape[0])

        for i in range(output[3].shape[0]):
            pose = output[4][i]
            _, height, width = output[2][i].shape

            #left hand
            lhandlocation = np.array([pose[self.lefthand*2],pose[self.lefthand*2+1]])*np.array([height, width])
            lhandlocation = lhandlocation.astype(int)
            lhandrgb, lhandflow = self.get_box_and_fill(lhandlocation-self.box_length,lhandlocation+self.box_length,output[2][i],output[3][i])
            lhandrgbv.append(transformer_RGB(lhandrgb) - self.rgb_avg)
            lhandflowv.append(transformer_Flow(lhandflow) - self.flow_avg)

            #right hand
            rhandlocation = np.array([pose[self.righthand*2],pose[self.righthand*2+1]])*np.array([height, width])
            rhandlocation = rhandlocation.astype(int)
            rhandrgb, rhandflow = self.get_box_and_fill(rhandlocation-self.box_length,rhandlocation+self.box_length,output[2][i],output[3][i])
            rhandrgbv.append(transformer_RGB(rhandrgb) - self.rgb_avg)
            rhandflowv.append(transformer_Flow(rhandflow) - self.flow_avg)

            #upperbody
            upperpose_x = [pose[ind*2] for ind in self.upperbody]
            upperpose_y = [pose[ind*2+1] for ind in self.upperbody]
            upperpose_x_nonzero = [upperpose_x[ind] for ind in list(np.nonzero(upperpose_x)[0])]
            upperpose_y_nonzero = [upperpose_y[ind] for ind in list(np.nonzero(upperpose_y)[0])]
            if len(upperpose_x_nonzero) == 0 or len(upperpose_y_nonzero) == 0:
                uppertopleft = np.array([0,0]).astype(int)
                upperbotright = np.array([0,0]).astype(int)
            else:
                uppertopleft = np.array([min(upperpose_x_nonzero),min(upperpose_y_nonzero)])*np.array([height, width])
                upperbotright = np.array([max(upperpose_x_nonzero),max(upperpose_y_nonzero)])*np.array([height, width])
                uppertopleft = uppertopleft.astype(int)
                upperbotright = upperbotright.astype(int)
            upperbodyrgb, upperbodyflow = self.get_box_and_fill(uppertopleft-self.box_length,upperbotright+self.box_length,output[2][i],output[3][i])
            upperbodyrgbv.append(transformer_RGB(upperbodyrgb) - self.rgb_avg)
            upperbodyflowv.append(transformer_Flow(upperbodyflow) - self.flow_avg)

            #full body
            fullbody_x = [pose[ind*2] for ind in self.fullbody]
            fullbody_y = [pose[ind*2+1] for ind in self.fullbody]
            fullbody_x_nonzero = [fullbody_x[ind] for ind in list(np.nonzero(fullbody_x)[0])]
            fullbody_y_nonzero = [fullbody_y[ind] for ind in list(np.nonzero(fullbody_y)[0])]
            if len(fullbody_x_nonzero) == 0 or len(fullbody_y_nonzero) == 0:
                fulltopleft = np.array([0,0]).astype(int)
                fullbotright = np.array([0,0]).astype(int)
            else:
                fulltopleft = np.array([min(fullbody_x_nonzero),min(fullbody_y_nonzero)])*np.array([height, width])
                fullbotright = np.array([max(fullbody_x_nonzero),max(fullbody_y_nonzero)])*np.array([height, width])
                fulltopleft = fulltopleft.astype(int)
                fullbotright = fullbotright.astype(int)
            fullbodyrgb, fullbodyflow = self.get_box_and_fill(fulltopleft-self.box_length,fullbotright+self.box_length,output[2][i],output[3][i])
            fullbodyrgbv.append(transformer_RGB(fullbodyrgb) - self.rgb_avg)
            fullbodyflowv.append(transformer_Flow(fullbodyflow) - self.flow_avg)
            #full image
            fullimagergbv.append(transformer_RGB(output[2][i]) - self.rgb_avg)
            fullimageflowv.append(transformer_Flow(output[3][i]) - self.flow_avg)
        
        lhandrgbv = torch.stack(lhandrgbv, 0)
        lhandflowv = torch.stack(lhandflowv, 0)
        rhandrgbv = torch.stack(rhandrgbv, 0)
        rhandflowv = torch.stack(rhandflowv, 0)
        upperbodyrgbv = torch.stack(upperbodyrgbv, 0)
        upperbodyflowv = torch.stack(upperbodyflowv, 0)
        fullbodyrgbv = torch.stack(fullbodyrgbv, 0)
        fullbodyflowv = torch.stack(fullbodyflowv, 0)
        fullimagergbv = torch.stack(fullimagergbv, 0)
        fullimageflowv = torch.stack(fullimageflowv, 0)

        # print(fullimagergbv.shape)

        return output[0],output[1],lhandrgbv,lhandflowv,rhandrgbv,rhandflowv,upperbodyrgbv,upperbodyflowv,fullbodyrgbv,fullbodyflowv,fullimagergbv,fullimageflowv

    def __len__(self):
        return len(self.n_sample_ind_iter)