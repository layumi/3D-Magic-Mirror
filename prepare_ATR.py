import glob
import os
from PIL import Image, ImageFilter, ImageOps
import numpy as np
# You only need to change this line to your dataset download path
np.random.seed(0)

download_path = '../ATR/humanparsing/SegmentationClassAug' 

seg_list = glob.glob(os.path.join(download_path,'*.png'))

percentage_list = []
new_seg_list = []
for img_path in seg_list:
    seg = Image.open(img_path).convert('L')
    seg = seg.point(lambda p: p > 0 and 255)
    foreground = np.sum( np.array(seg)/255 )
    percentage = foreground/(seg.size[0]*seg.size[1])
    new_name = img_path.replace('.png','_%.2f.png'%percentage)
    print(img_path, new_name)
    os.system('mv %s %s'%( img_path, new_name) )
    percentage_list.append(percentage)
    new_seg_list.append(new_name)

print(sum(percentage_list)/len(percentage_list))
rand_index = np.random.permutation(len(new_seg_list))
print(rand_index)
train_list = open('./datasets/ATR_train.txt','a') # 16000
test_list = open('./datasets/ATR_test.txt','a') # 1606
for index in rand_index[0:16000]:
    train_list.writelines(os.path.basename(new_seg_list[index])+'\n')

for index in rand_index[16000:]:
    test_list.writelines(os.path.basename(new_seg_list[index])+'\n')

train_list.close()
test_list.close()
