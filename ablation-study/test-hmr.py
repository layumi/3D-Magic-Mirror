from PIL import Image,ImageFilter, ImageOps
from pytorch_msssim import ssim
from kaolin.metrics.render import mask_iou
import torchvision
import numpy as np
import torch

target_height, target_width  = 128, 128
# Recon SSIM
ssim_score = []
mask_score = []
with open('../datasets/ATR_test.txt', 'r') as f:
    test_list = [line.strip() for line in f]

ori_dir = '../../ATR/humanparsing/JPEGImages'
ori_mask_dir = '../../ATR/humanparsing/SegmentationClassAug'
rec_dir = '../../hmr/3DATR_hmr_mask'

for name in test_list:
    # SSIM
    ori_path = ori_dir + '/' + name
    seg_path = ori_mask_dir + '/' + name.replace('.jpg','.png')
    rec_path = rec_dir + '/' + name + '.png'

    img = Image.open(ori_path).convert('RGB')
    seg = Image.open(seg_path).convert('L')
    rec = Image.open(rec_path).convert('L')

    W, H = img.size
    desired_size = max(W, H)
    delta_w = desired_size - W
    delta_h = desired_size - H
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    img = ImageOps.expand(img, padding)
    seg = ImageOps.expand(seg, padding).point(lambda p: p > 0 and 255)


    W, H = rec.size
    desired_size = max(W, H)
    delta_w = desired_size - W
    delta_h = desired_size - H
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    rec = ImageOps.expand(rec, padding).point(lambda p: p > 0 and 255)

    img = img.resize((target_height, target_width))
    seg = seg.resize((target_height, target_width), Image.NEAREST)
    seg = seg.point(lambda p: p > 160 and 255)
    rec = rec.resize((target_height, target_width), Image.NEAREST)
    rec = rec.point(lambda p: p > 160 and 255)

    img = torchvision.transforms.functional.to_tensor(img)
    seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]
    rec_mask = torchvision.transforms.functional.to_tensor(rec).max(0, True)[0]


    ori = img * seg + torch.ones_like(img) * (1 - seg)
    rec = img * rec_mask + torch.ones_like(img) * (1 - rec_mask)
    ssim_score.append(ssim(ori.unsqueeze(0), rec.unsqueeze(0), data_range=1))
    mask_score.append(1 - mask_iou(seg, rec_mask)) # the default mask iou is maskiou loss. https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/render.py. So we have to 1- maskiou loss to obtain the mask iou

print('\033[1mTest recon ssim: %0.3f \033[0m' % np.mean(ssim_score) )
print('\033[1mTest recon MaskIoU: %0.3f\033[0m' % np.mean(mask_score) )

