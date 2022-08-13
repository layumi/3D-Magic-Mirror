import glob
import os
#import accimage
from PIL import Image, ImageFilter, ImageOps
import numpy as np
# You only need to change this line to your dataset download path
download_path = './data/CUB_Data/' 

seg_list = glob.glob(os.path.join(download_path, 'train', '*/*.png'))

for img_path in seg_list:
    seg = Image.open(img_path).convert('RGB')
    #img2 = accimage.Image(img_path) 
    #print(np.array(img)[0]) 
    seg = seg.point(lambda p: p > 160 and 255)
    seg.save(img_path.replace('.png','_smooth.png'))
    edge = seg.filter(ImageFilter.FIND_EDGES)
    edge = edge.filter(ImageFilter.SMOOTH_MORE)
    edge = edge.point(lambda p: p > 20 and 255)
    edge.save(img_path.replace('.png','_edge.png'))


    w, h = seg.width, seg.height
    coarse_edge = np.asarray(seg) - np.asarray(seg.resize((w//2,h//2)).resize((w, h)))
    coarse_edge = Image.fromarray(np.uint8(np.abs(coarse_edge)))
    coarse_edge.save(img_path.replace('.png','_coarse_edge.png'))

    print(img_path)
    break

