import glob
import os
from PIL import Image, ImageFilter, ImageOps
import numpy as np
# You only need to change this line to your dataset download path
np.random.seed(0)

download_path = '../ATR/humanparsing/SegmentationClassAug' 

os.makedirs('../ATR/humanparsing/Seg', exist_ok=True)
os.system('rm -r ../ATR/humanparsing/Seg/*')
seg_list = glob.glob(os.path.join(download_path,'*.png'))

percentage_list = []
new_seg_list = {}
for img_path in seg_list:
    seg = Image.open(img_path).convert('L')
    seg = seg.point(lambda p: p > 0 and 255)
    foreground = np.sum( np.array(seg)/255 )
    percentage = foreground/(seg.size[0]*seg.size[1])
    new_name = img_path.replace('.png','_%.2f.png'%percentage)
    new_name = new_name.replace('SegmentationClassAug', 'Seg')
    print(img_path, new_name)
    os.system('cp %s %s'%( img_path, new_name) )
    percentage_list.append(percentage)
    new_seg_list[os.path.basename(img_path)] = os.path.basename(new_name)
    
print(sum(percentage_list)/len(percentage_list))

test_list = open('./datasets/ATR_test.txt','w') # 1606
train_list = open('./datasets/ATR_train.txt','w') # 16000

with open('./datasets/ATR_train_old.txt') as f:
    lines = f.readlines()
    for old_name in lines:
        old_name = old_name.rstrip().replace('jpg', 'png')
        train_list.writelines(os.path.basename(new_seg_list[old_name])+'\n')

with open('./datasets/ATR_test_old.txt') as f:
    lines = f.readlines()
    for old_name in lines:
        old_name = old_name.rstrip().replace('jpg', 'png')
        test_list.writelines(os.path.basename(new_seg_list[old_name])+'\n')

train_list.close()
test_list.close()
