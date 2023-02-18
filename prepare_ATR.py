import glob
import os
from PIL import Image, ImageFilter, ImageOps
from numba import jit
import numpy as np
import torchvision
import torch
# You only need to change this line to your dataset download path
np.random.seed(0)

download_path = '../ATR/humanparsing/SegmentationClassAug' 

os.makedirs('../ATR/humanparsing/Seg', exist_ok=True)
os.system('rm -r ../ATR/humanparsing/Seg/*')
seg_list = glob.glob(os.path.join(download_path,'*.png'))
percentage_list = []
new_seg_list = {}

meanpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

for img_path in seg_list:
    seg = Image.open(img_path).convert('L')
    seg = seg.point(lambda p: p > 0 and 255)

    w, h = seg.size
    seg = torchvision.transforms.functional.pil_to_tensor(seg).float()
    # make up holes. 
    for t in range(5):
        seg = seg + meanpool(seg)
        seg[seg > (4/9)] = 1
        seg[seg <= (4/9)] = 0

    foreground = torch.sum(seg )
    percentage = foreground/(h*w)
    print(percentage)
    new_name = img_path.replace('.png','_%.2f.png'%percentage)
    new_name = new_name.replace('SegmentationClassAug', 'Seg')
    print(img_path, new_name)
    #os.system('cp %s %s'%( img_path, new_name) )
    seg = torchvision.transforms.functional.to_pil_image(seg)
    seg.save(new_name)
    percentage_list.append(percentage)
    new_seg_list[os.path.basename(img_path)] = os.path.basename(new_name)
    
print(sum(percentage_list)/len(percentage_list))

#test_list = open('./datasets/ATR_test.txt','w') # 1606
#train_list = open('./datasets/ATR_train.txt','w') # 16000

#with open('./datasets/ATR_train_old.txt') as f:
#    lines = f.readlines()
#    for old_name in lines:
#        old_name = old_name.rstrip().replace('jpg', 'png')
#        train_list.writelines(os.path.basename(new_seg_list[old_name])+'\n')

#with open('./datasets/ATR_test_old.txt') as f:
#    lines = f.readlines()
#    for old_name in lines:
#        old_name = old_name.rstrip().replace('jpg', 'png')
#        test_list.writelines(os.path.basename(new_seg_list[old_name])+'\n')

#train_list.close()
#test_list.close()
