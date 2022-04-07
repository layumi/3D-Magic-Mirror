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
#import open3d as o3d
import kaolin as kal
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
    def __init__(self, root, image_size, transform=None, loader=default_loader, train=True, return_paths=False, threshold=0.09, bg=False, hmr = 0.0, selected_index = []):
        super(MarketDataset, self).__init__()
        self.root = root
        self.bg = bg
        self.hmr = hmr
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
            if precentage>threshold and precentage<0.64:
                self.im_list.append(name)
        print(len(old_im_list),'After threshold:',len(self.im_list))
        #self.im_list = old_im_list
        self.transform = transform
        self.loader = loader
        self.seg_loader = seg_loader

        # sort im_list
        #self.im_list = sorted(self.im_list)
        self.imgs = [(im_path, self.class_dir.index(os.path.dirname(im_path))) for
                     im_path in self.im_list]
        #random.shuffle(self.imgs)

        self.return_paths = return_paths
        self.train = train
        self.image_size = image_size
        self.selected_index = selected_index
        print('Succeed loading dataset!')

    def __getitem__(self, index):
        if len(self.selected_index) > 0: # for test
            index = self.selected_index[index]
        seg_path, label = self.imgs[index]
        target_width = self.image_size

        # image and its flipped image
        #img_path = seg_path.replace('seg', 'pytorch')
        img_path = seg_path.replace('seg_hmr', 'pytorch')
        # remove foreground precentage
        img_path = img_path[:-9] + '.png'
        img = self.loader(img_path)
        seg = self.seg_loader(seg_path)
        W, H = img.size
        if self.hmr>0.0:
            obj_path = seg_path.replace('seg_hmr', 'bodymesh')
            obj_path = obj_path[:-9] + '.obj'
            mesh =  kal.io.obj.import_mesh(obj_path)
            obj = np.asarray(mesh.vertices, dtype=np.float32) # 6890*3
        else:
            obj = -1

        if self.train:
            # resize 128x64 (the effective part is 128x64)
            #ratio_h = random.uniform(1.0, 1.1)
            #ratio_w = random.uniform(1.0, 1.1)
            #img = img.resize((int(target_width*ratio_w), int(target_width*ratio_h*2)))
            #seg = seg.resize((int(target_width*ratio_w), int(target_width*ratio_h*2)), Image.NEAREST)
            img = img.resize((target_width, target_width*2))
            seg = seg.resize((target_width, target_width*2), Image.NEAREST)
            seg = seg.point(lambda p: p > 160 and 255)

            #padding 10
            #img = torchvision.transforms.functional.pad(img, 10, 0, "constant")
            #seg = torchvision.transforms.functional.pad(seg, 10, 0, "constant")
            img = ImageOps.expand(img, 10)
            seg = ImageOps.expand(seg, 10)
            # random crop
            h, w = target_width*2, target_width 
            left = random.randint(0, 20)
            upper = random.randint(0, 20)
            img = img.crop((left, upper, left+w, upper+h))
            seg = seg.crop((left, upper, left+w, upper+h))

            # random flip
            if random.uniform(0, 1) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
                if self.hmr>0.0:
                    obj[:,0] *= -1 # note this obj need to be normalized before we can use. More code is needed.

        img = img.resize((int(target_width), int(target_width*2)))
        seg = seg.resize((int(target_width), int(target_width*2)), Image.NEAREST)
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

        data= {'images': rgbs, 'path': img_path, 'label': label, 'rgb': rgb, 'obj': obj} #,
              # 'edge': edge}

        return {'data': data}

    def __len__(self):
        return len(self.imgs)
