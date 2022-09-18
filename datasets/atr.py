"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image, ImageFilter, ImageOps

import torch.utils.data as data
import torch
import torchvision
import numpy as np
import random
#import accimage # conda install -c conda-forge accimage

def default_loader(path):
    #return accimage.Image(path)#.convert('RGB')
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def seg_loader(path):
    with open(path, 'rb') as f:
        seg = Image.open(f).convert('L')
        seg = seg.point(lambda p: p > 0 and 255)
        return seg

class ATRDataset(data.Dataset):
    def __init__(self, root, image_size, transform=None, loader=default_loader, train=True, return_paths=False, bg=False, selected_index = []):
        super(ATRDataset, self).__init__()
        self.root = root
        self.bg = bg
        if train:
            with open('datasets/ATR_train.txt', 'r') as f:
                self.im_list = [root+'/'+line.strip() for line in f]
        else:
            with open('datasets/ATR_test.txt', 'r') as f:
                self.im_list = [root+'/'+line.strip() for line in f]

        self.transform = transform
        self.loader = loader
        self.seg_loader = seg_loader
        self.imgs = [(im_path, -1)  for
                     im_path in self.im_list]  # no class label
        #random.shuffle(self.imgs)

        self.return_paths = return_paths
        self.train = train
        self.image_size = image_size
        self.selected_index = selected_index
        print('Succeed loading dataset!')

    def __getitem__(self, index):
        if len(self.selected_index) >0:
            original_index = index
            index = self.selected_index[index]
 
        img_path, label = self.imgs[index]
        if len(self.selected_index) >0:
            print(original_index, img_path)
        target_height, target_width = self.image_size, self.image_size

        # image and its flipped image
        seg_path = img_path.replace('JPEGImages','SegmentationClassAug').replace('.jpg', '.png')
        img = self.loader(img_path)
        seg = self.seg_loader(seg_path)
        W, H = img.size

        if self.train:
            if random.uniform(0, 1) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
            # pad 10
            img = ImageOps.expand(img, 10)
            seg = ImageOps.expand(seg, 10)
            W, H = W + 20, H+20
            # random crop mask & img
            w = random.randint(int(0.95 * W), int(0.99 * W))
            h = random.randint(int(0.95 * H), int(0.99 * H))
            left = random.randint(0, W-w)
            upper = random.randint(0, H-h)
            right = random.randint(w - left, W)
            lower = random.randint(h - upper, H)
            img = img.crop((left, upper, right, lower))
            seg = seg.crop((left, upper, right, lower))

        W, H = img.size
        desired_size = max(W, H)
        delta_w = desired_size - W
        delta_h = desired_size - H
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        seg = ImageOps.expand(seg, padding)

        img = img.resize((target_height, target_width))
        seg = seg.resize((target_height, target_width), Image.NEAREST)
        seg = seg.point(lambda p: p > 160 and 255)

        #edge = seg.filter(ImageFilter.FIND_EDGES)
        #edge = edge.filter(ImageFilter.SMOOTH_MORE)
        #edge = edge.point(lambda p: p > 20 and 255)
        #edge = torchvision.transforms.functional.to_tensor(edge).max(0, True)[0]

        img = torchvision.transforms.functional.to_tensor(img)
        seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]

        if self.bg:
            rgb = img
        else:
            rgb = img * seg + torch.ones_like(img) * (1 - seg)
        rgbs = torch.cat([rgb, seg], dim=0)

        data= {'images': rgbs, 'path': img_path, 'label': label} #,
              # 'edge': edge}        

        return {'data': data}

    def __len__(self):
        return len(self.imgs)
