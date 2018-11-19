import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class RGB2BGR(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return torch.from_numpy(np.ascontiguousarray(inputs.numpy()[::-1, :, :]))

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


def default_get_frame_inds(n_frame, n_seg, seq_strides, seq_len):
    if n_frame <= 0:
        return np.array([0]*(n_seg*seq_len))
    assert n_frame > 0
    rand_stride = random.choice(seq_strides)
    # use linspace to set boundaries of segs
    end_points = np.linspace(0, n_frame - 1, n_seg + 1).round().astype(np.int).tolist()

    frame_inds = []
    for seg_ind in range(n_seg):
        if rand_stride == 'center_random':
            assert n_seg == 3
            if seg_ind != 1:
                continue
        seg_start = end_points[seg_ind]
        seg_end = end_points[seg_ind+1]
        if isinstance(rand_stride, int):
            strided_seg = range(seg_start, seg_end + 1, rand_stride)  # at least one sample here
        elif rand_stride == 'random' or rand_stride == 'center_random':
            strided_seg = sorted(np.unique(np.random.randint(seg_start, seg_end+1, seq_len)).tolist())
        elif rand_stride == 'uniform':
            strided_seg = np.arange(seg_start, seg_end+1)
            seg_len = len(strided_seg)
            strided_seg = strided_seg[np.linspace(0, seg_len - 1, seq_len).round().astype(np.int).tolist()].tolist()
        elif rand_stride == 'center':
            strided_seg = np.arange(seg_start, seg_end+1)
            if len(strided_seg) > seq_len:
                offset = (len(strided_seg) - seq_len) / 2
                strided_seg = strided_seg[offset:offset+seq_len]
            strided_seg = strided_seg.tolist()

        else:
            raise ValueError('stride mode wrong')

        rand_start = random.randrange(0, max(len(strided_seg) - seq_len + 1, 1))
        sample_inds = strided_seg[rand_start:rand_start+seq_len]
        if len(sample_inds) < seq_len:
            sample_inds = np.pad(sample_inds, (0, seq_len - len(sample_inds)), mode='wrap')
            sample_inds = sample_inds.tolist()
        frame_inds += sample_inds
    frame_inds = np.asarray(frame_inds)
    return frame_inds


import cv2

def video_loader(video_path, frame_inds):
    vid = cv2.VideoCapture(video_path)
    ind = 0
    i = 0
    rgb_np = []
    sorted_frame_inds = np.sort(np.unique(frame_inds))

    while True:
        flag = vid.grab()
        if not flag:
            break
        if ind in sorted_frame_inds:
            flag, img = vid.retrieve()
            assert len(img.shape) == 3
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_np.append(img)
            i += 1
        ind += 1
    ind_list = np.searchsorted(sorted_frame_inds, frame_inds)
    output = [Image.fromarray(rgb_np[i].copy(), mode='RGB') for i in ind_list]
    vid.release()
    return output

def video_loader_transfomer(video_path, frame_inds, transformer, config):
    vid = cv2.VideoCapture(video_path)
    ind = 0
    i = 0
    rgb_np = []
    sorted_frame_inds = np.sort(np.unique(frame_inds))
    if transformer is None:
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            #  torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor()
        ])
    else:
        for t in transformer.transforms:
            if hasattr(t, 'set_rnd'):
                t.set_rnd()

    while True:
        flag = vid.grab()
        if not flag:
            break
        if ind in sorted_frame_inds:
            flag, img = vid.retrieve()
            assert len(img.shape) == 3
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_np.append(transformer(img))
            i += 1
        ind += 1
    if len(rgb_np) != len(sorted_frame_inds):
        output = torch.zeros(len(frame_inds), 3, 224, 224)
        print('wrong loading! {}'.format(video_path))
    else:
        ind_list = np.searchsorted(sorted_frame_inds, frame_inds)
        output = torch.stack([rgb_np[i].clone() for i in ind_list], 0)
    vid.release()
    return output
import h5py
import bisect
import os
def jpg_loader(vid_path, frame_inds, transformer, config):
    rgb_np = []
    sorted_frame_inds = np.sort(np.unique(frame_inds))
    if transformer is None:
        # transformer = torchvision.transforms.ToTensor()
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            #   torchvision.transforms.Resize([224,224]),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    else:
        for t in transformer.transforms:
            if hasattr(t, 'set_rnd'):
                t.set_rnd()
    if 'jpg_format' not in config:
        jpg_format = 'img_{:06d}.jpg'
    else:
        jpg_format = config['jpg_format']
    if '{vid_name}' in jpg_format:
        vid_name = os.path.dirname(vid_path)
    else:
        vid_name = ""

    for frame_ind in sorted_frame_inds:
        img_path = os.path.join(vid_path, jpg_format.format(frame_ind+1, vid_name=vid_name))
        # print(img_path)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            assert len(img.shape) == 3
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
            # print(img_path, 'not found')
            # raise ValueError
            # img = np.zeros((256, 340, 3), np.uint8)
        # print(img.shape)
            rgb_np.append(transformer(img))
    ind_list = np.searchsorted(sorted_frame_inds, frame_inds)

    # output = torch.stack([rgb_np[i].clone() for i in ind_list], 0)
    try:
        output = torch.stack(rgb_np, 0)
    except RuntimeError:
        print(frame_inds)
        print(img_path)
        print(rgb_np)
        exit()
    return output

def flow_loader(vid_path, frame_inds, transformer, config):
    flow = []
    sorted_frame_inds = np.sort(np.unique(frame_inds))
    if transformer is None:
        # transformer = torchvision.transforms.ToTensor()
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Resize([227,227]),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485],std=[0.229])
        ])
    else:
        for t in transformer.transforms:
            if hasattr(t, 'set_rnd'):
                t.set_rnd()
    if 'jpg_format' not in config:
#        jpg_format_x = 'flow_x/x_{:05d}.jpg'    #UCF
#        jpg_format_y = 'flow_y/y_{:05d}.jpg'    #UCF
         jpg_format_x = 'flow_x/frame{:06d}.jpg'    #HMDB
         jpg_format_y = 'flow_y/frame{:06d}.jpg'    #HMDB
    else:
        jpg_format = config['jpg_format']
    if '{vid_name}' in jpg_format_x:
        vid_name = os.path.dirname(vid_path)
    else:
        vid_name = ""
    if sorted_frame_inds[-1] != 0:
        sorted_frame_inds[-1] = sorted_frame_inds[-1] - 1
    # print(sorted_frame_inds)
    for frame_ind in sorted_frame_inds:
        imgs=[]
        img_path_x = os.path.join(vid_path, jpg_format_x.format(frame_ind+1, vid_name=vid_name))
        img_path_y = os.path.join(vid_path, jpg_format_y.format(frame_ind+1, vid_name=vid_name))
        if os.path.exists(img_path_x):
            img_x = cv2.imread(img_path_x,0)
            img_y = cv2.imread(img_path_y,0)
            # img_mag = np.sqrt(img_x**2+img_y**2)
            assert len(img_x.shape) == 2
        # else:
            # print(img_path, 'not found')
            # raise ValueError
            # img = np.zeros((256, 340, 3), np.uint8)
        # print(img.shape)
            imgs.append(transformer(np.expand_dims(img_x,2))[0])
            imgs.append(transformer(np.expand_dims(img_y,2))[0])
            imgs.append(torch.sqrt(imgs[0]**2+imgs[1]**2))
            imgs = torch.stack(imgs,0)
            flow.append(imgs)
    ind_list = np.searchsorted(sorted_frame_inds, frame_inds)
    # output = torch.stack([rgb_np[i].clone() for i in ind_list], 0)
    try:
        output = torch.stack(flow, 0)
    except RuntimeError:
        print(frame_inds)
        print(img_path_x)
        print(flow)
        exit()
    # if(output.size()[0] == 2):
    #     print(sorted_frame_inds)
    # print(output.size())
    return output

def h5_loader(vid_path, frame_inds, t, config):#**kwargs):
    with h5py.File(vid_path, 'r') as f:
        if len(f['ind'][()]) == 1 :
          inds = f['ind'][()][0] #changed for somethingsomething dataset
        else:
          inds = f['ind'][()]
          
        to_read_inds = []
        for r in frame_inds:
            r = bisect.bisect_left(inds, r)
            if r == len(inds):
               r = len(inds) - 1
            # if r > len(inds):
            #     r = len(inds) - 1
            to_read_inds.append(r)
        sorted_frame_inds = np.sort(np.unique(to_read_inds))

        t_features = f['feature'][sorted_frame_inds.tolist()]
        features = t_features[np.searchsorted(sorted_frame_inds, to_read_inds)]
    return features

def img_loader(img_path, **kwargs):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

import torch.multiprocessing as mp
import threading
import queue
def _video_reader_worker(transformer, vid_path, data_queue, batch_size, stride, seq_len):
    torch.set_num_threads(1)
    vid = cv2.VideoCapture(vid_path)

    frame_ind = 0

    cur_frames = []
    frame_inds = []
    while True:

        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur_frames.append(transformer(frame))
        frame_inds.append(frame_ind)
        frame_ind += 1
        if frame_ind == seq_len:
            break

    if len(cur_frames) > 0:
        t_cur_frames = torch.zeros((seq_len, ) + cur_frames[0].size())
    else:
        raise

    for i in range(len(cur_frames)):
        t_cur_frames[i].copy_(cur_frames[i])
    cur_frames = t_cur_frames
    frame_inds_batch = []
    frames_batch = []
    frames_batch.append(cur_frames)
    frame_inds_batch.append(frame_inds)

    while True:
        # batch iter
        if len(frames_batch) == batch_size:
            frames_batch = torch.stack(frames_batch, 0)
            data_queue.put([frame_inds_batch, frames_batch])
            frames_batch = []
            frame_inds_batch = []
        # seg iter
        breakout = False
        if seq_len > stride:
            cur_frames = [cur_frames[stride:]]
            frame_inds = frame_inds[stride:]
            n_cur = cur_frames[0].size(0)
        else:
            cur_frames = []
            frame_inds = []
            n_cur = 0
        # skip frames
        for _ in range(stride - seq_len):
            ret = vid.grab()

            if not ret:
                breakout = True
                break
            frame_ind += 1
        for _ in range(seq_len - n_cur):
            ret = vid.grab()

            if not ret:
                breakout = True
                break

            ret, frame = vid.retrieve()
            if not ret:
                breakout = True
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur_frames.append(transformer(frame).unsqueeze(0))
            frame_inds.append(frame_ind)
            frame_ind += 1

        if len(cur_frames) > 0:
            cur_frames = torch.cat(cur_frames, 0)

            if len(cur_frames) < seq_len:
                shape = list(cur_frames.size())
                shape[0] = seq_len - len(cur_frames)
                shape = tuple(shape)
                cur_frames = torch.cat([cur_frames, cur_frames.new(*shape).zero_()], 0)
                frame_inds.extend([-1]*(seq_len - len(cur_frames)))

            frames_batch.append(cur_frames)
            frame_inds_batch.append(frame_inds)
            #print(cur_frames.size())
        else:
            break

        if breakout:
            break
    if len(frames_batch) != 0:
        #print([ f.size() for f in frames_batch])
        frames_batch = torch.stack(frames_batch, 0)

        data_queue.put([frame_inds_batch, frames_batch])
    vid.release()

class AsyncVideoReader(object):
    def __init__(self, transformer, batch_size, stride, seq_len):
        self.transformer = transformer

        self.data_queue = queue.Queue(50) #mp.Queue(50)
        self.batch_size = batch_size
        self.stride = stride
        self.seq_len = seq_len

    def start_reader(self, vid_path):
        # w = mp.Process(target=_video_reader_worker, args=(self.transformer, vid_path, self.data_queue, self.batch_size,
        #                                                   self.stride, self.seq_len))
        w = threading.Thread(target=_video_reader_worker, args=(self.transformer, vid_path, self.data_queue, self.batch_size,
                                                          self.stride, self.seq_len))
        w.start()
        while True:
            try:
                data = self.data_queue.get(True, 10.0)
                yield data
            except Exception as e:
                print(e)
                break
        w.join()
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get(True, 10.0)
                yield data
            except Exception as e:
                print(e)
                break


def test_video_loader():
    from torchvision import transforms
    video_path = '/data/yingwei/vidDB/UCF101/ori_data/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        # ScaleRect(340, 256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5,], std=[1.0, ])
    ])

    imgs = video_loader(video_path, range(250), )
    for i, img in enumerate(imgs):
        img_vis = img.numpy().transpose(1, 2, 0)[:, :, ::-1]

        cv2.imshow('img', img_vis)
        cv2.waitKey(30)

def test_async_video_loader():
    from torchvision import transforms
    transformer = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    reader = AsyncVideoReader(transformer, 32, 4, 16)
    video_path = '/data/yingwei/vidDB/UCF101/ori_data/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    for inds, batch in reader.start_reader(video_path):
        #print(inds, batch)
        imgs = batch[0]

        for i, img in enumerate(imgs):
            #cv2.imwrite('img_{}.jpg'.format(i), (img.numpy().transpose(1, 2, 0)*255).astype(np.uint8))
            #cv2.waitKey(10)
            pass


if __name__ == '__main__':
    test_async_video_loader()