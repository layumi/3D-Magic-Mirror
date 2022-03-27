import glob
import os
import accimage
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch
import torchvision
# You only need to change this line to your dataset download path
img_path = './data/CUB_Data/test/022.Chuck_will_Widow/Chuck_Will_Widow_0051_796991.jpg' 
seg_path = './data/CUB_Data/test/022.Chuck_will_Widow/Chuck_Will_Widow_0051_796991.png' 
#seg_path = './data/CUB_Data/test/022.Chuck_will_Widow/Chuck_Will_Widow_0022_796967.png' 

img = Image.open(img_path)
seg = Image.open(seg_path).convert('L')
print(seg.size)
print(seg)
seg = seg.point(lambda p: p > 160 and 255)

img = torchvision.transforms.functional.to_tensor(img)
seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]

print(seg)
rgb = img * seg + torch.ones_like(img) * (1 - seg)
rgb = Image.fromarray(rgb.data.numpy())
rgb.save('./seg_demo.png')

