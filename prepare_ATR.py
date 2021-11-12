import glob
import os
from PIL import Image, ImageFilter, ImageOps
import numpy as np
# You only need to change this line to your dataset download path
np.random.seed(0)

download_path = '../ATR/humanparsing/JPEGImages' 

seg_list = glob.glob(os.path.join(download_path,'*.jpg'))

rand_index = np.random.permutation(len(seg_list))
print(rand_index)
train_list = open('./datasets/ATR_train.txt','a') # 16000
test_list = open('./datasets/ATR_test.txt','a') # 1606
for index in rand_index[0:16000]:
    train_list.writelines(os.path.basename(seg_list[index])+'\n')

for index in rand_index[16000:]:
    test_list.writelines(os.path.basename(seg_list[index])+'\n')

train_list.close()
test_list.close()
