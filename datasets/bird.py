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
        seg = seg.point(lambda p: p > 160 and 255)
        return seg

class CUBDataset(data.Dataset):
    def __init__(self, root, image_size, transform=None, loader=default_loader, train=True, aug=False, return_paths=False, threshold='0.09,0.64', bg = False, selected_index = []):
        super(CUBDataset, self).__init__()
        self.root = root
        self.bg = bg
        self.im_list=[]
        if train:
            old_im_list = glob.glob(os.path.join(self.root, 'train', '*/*.png'))
            self.class_dir = glob.glob(os.path.join(self.root, 'train', '*'))
        else:
            old_im_list = sorted(glob.glob(os.path.join(self.root, 'test', '*/*.png')))
            self.class_dir = glob.glob(os.path.join(self.root, 'test', '*'))

        # threshold
        threshold = threshold.replace(' ','').split(',')
        for index, name in enumerate(old_im_list):
            precentage = float(name[-8:-4])
            if precentage>float(threshold[0]) and precentage<float(threshold[1]):
                self.im_list.append(name)
        if not train:
            self.im_list = old_im_list
        print(len(old_im_list),'After threshold:',len(self.im_list))

        self.transform = transform
        self.loader = loader
        self.seg_loader = seg_loader

        self.imgs = [(im_path, self.class_dir.index(os.path.dirname(im_path))) for
                     im_path in self.im_list]
        #random.shuffle(self.imgs)

        self.return_paths = return_paths
        self.train = train
        self.aug = aug
        self.image_size = image_size

        self.selected_index = selected_index
        print('Succeed loading dataset!')

    def __getitem__(self, index):
        if len(self.selected_index) >0:
            index = self.selected_index[index]
        index = index%len(self.imgs)
        seg_path, label = self.imgs[index]
        target_height, target_width = self.image_size, self.image_size

        # image and its flipped image
        #seg_path = img_path.replace('.jpg', '.png')
        #img_path = seg_path.replace('.png', '.jpg')
        img_path = seg_path[:-9] + '.jpg'
        img = self.loader(img_path)
        seg = self.seg_loader(seg_path) # Pillow Image Behavior is not stable. So the convert is neccessary. 
        W, H = img.size
        if self.train and self.aug:
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
            left = random.randint(0, W-w) # note randomint is [0, W-w]
            upper = random.randint(0, H-h)
            right = random.randint(w - left, W)
            lower = random.randint(h - upper, H)
            img = img.crop((left, upper, right, lower))
            seg = seg.crop((left, upper, right, lower))
            
            #img = np.asarray(img, np.uint8)
            #seg = np.asarray(seg, np.uint8)
            #img = img[left:right, upper:lower, :]
            #seg= seg[left:right, upper:lower]
            #img = Image.fromarray(np.uint8(img))
            #seg = Image.fromarray(np.uint8(seg))

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

        data= {'images': rgbs, 'path': img_path, 'label': label}        

        return {'data': data}

    def __len__(self):
        return len(self.imgs)*2
