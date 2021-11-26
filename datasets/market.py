"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image, ImageFilter, ImageOps
import glob

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

class MarketDataset(data.Dataset):
    def __init__(self, root, image_size, transform=None, loader=default_loader, train=True, return_paths=False, threshold=0.36, bg=False):
        super(MarketDataset, self).__init__()
        self.root = root
        self.bg = bg
        self.im_list = []
        if train:
            old_im_list = glob.glob(os.path.join(self.root, 'train_all', '*/*.png'))
            self.class_dir = glob.glob(os.path.join(self.root, 'train_all', '*'))
        else:
            old_im_list = glob.glob(os.path.join(self.root, 'query', '*/*.png'))
            self.class_dir = glob.glob(os.path.join(self.root, 'query', '*'))

        # threshold
        for index, name in enumerate(old_im_list):
            precentage = float(name[-8:-4])
            if precentage>threshold and precentage<0.81:
                self.im_list.append(name)
        print(len(old_im_list),'After threshold:',len(self.im_list))

        self.transform = transform
        self.loader = loader
        self.seg_loader = seg_loader
        self.imgs = [(im_path, self.class_dir.index(os.path.dirname(im_path))) for
                     im_path in self.im_list]
        random.shuffle(self.imgs)

        self.return_paths = return_paths
        self.train = train
        self.image_size = image_size

        print('Succeed loading dataset!')

    def __getitem__(self, index):
        seg_path, label = self.imgs[index]
        target_height, target_width = self.image_size, self.image_size

        # image and its flipped image
        img_path = seg_path.replace('seg', 'pytorch')
        # remove foreground precentage
        img_path = img_path[:-9] + '.png'
        img = self.loader(img_path)
        seg = self.seg_loader(seg_path)
        W, H = img.size

        if self.train:
            if random.uniform(0, 1) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

            #h = random.randint(int(0.90 * H), int(0.99 * H))
            #w = random.randint(int(0.90 * W), int(0.99 * W))
            #left = random.randint(0, W-w)
            #upper = random.randint(0, H-h)
            #right = random.randint(w - left, W)
            #lower = random.randint(h - upper, H)
            #img = img.crop((left, upper, right, lower))
            #seg = seg.crop((left, upper, right, lower))

        ###### make a square 512x512
        #W, H = img.size
        #desired_size = max(W, H)
        #delta_w = desired_size - W
        #delta_h = desired_size - H
        #padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))

        #img = ImageOps.expand(img, padding)
        #seg = ImageOps.expand(seg, padding)

        # resize 128x64 (the effective part is 128x64)
        img = img.resize((target_height, target_width*2))
        seg = seg.resize((target_height, target_width*2))
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
