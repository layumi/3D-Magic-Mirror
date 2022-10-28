import glob
import os
from PIL import Image, ImageFilter, ImageOps
import numpy as np
# You only need to change this line to your dataset download path
download_path = './data/CUB_Data' 

seg_list = glob.glob(os.path.join(download_path,'*/*/*.png'))

percentage_list = []
for img_path in seg_list:
    seg = Image.open(img_path).convert('L')
    seg = seg.point(lambda p: p > 0 and 255)
    foreground = np.sum( np.array(seg)/255 ) 
    percentage = foreground/(seg.size[0]*seg.size[1]) 
    new_name = img_path.replace('.png','_%.2f.png'%percentage)
    print(img_path, new_name)
    os.system('mv %s %s'%( img_path, new_name) )
    percentage_list.append(percentage)
    #edge = seg.filter(ImageFilter.FIND_EDGES)
    #edge = edge.filter(ImageFilter.SMOOTH_MORE)
    #edge = edge.point(lambda p: p > 20 and 255)
    #edge.save(img_path.replace('.png','_edge.png'))
    #break

print(sum(percentage_list)/len(percentage_list))
