import os
import csv
import numpy as np
import skimage.io as sio
import skimage
from skimage.draw import circle
import tqdm
import cv2

root_dir = "./data/CUB_200_2011"
dst_dir = "./data/CUB_Data"

image_paths = np.loadtxt(os.path.join(root_dir, 'images.txt'), dtype=str, delimiter=' ')
image_class_labels = np.loadtxt(os.path.join(root_dir, 'image_class_labels.txt'), dtype=int, delimiter=' ')
train_test_split = np.loadtxt(os.path.join(root_dir, 'train_test_split.txt'), dtype=int, delimiter=' ')
bboxes = np.loadtxt(os.path.join(root_dir, 'bounding_boxes.txt'), dtype=float, delimiter=' ')
keypoints = np.loadtxt(os.path.join(root_dir, 'parts', 'part_locs.txt'), dtype=float, delimiter=' ')
keypoints = keypoints[:, 2:5].reshape([-1, 15, 3])

count = 0
for i in tqdm.tqdm(range(image_paths.shape[0])):
    image_path = os.path.join(root_dir, 'images', image_paths[i, 1])
    phase = 'train' if train_test_split[i, 1] else 'test'
    dst_path = os.path.join(dst_dir, phase, image_paths[i, 1])
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    rawImg = sio.imread(image_path)
    if len(rawImg.shape) < 3:
        rawImg = np.expand_dims(rawImg, 2)
    height, width = rawImg.shape[:2]

    seg_path = os.path.join(root_dir, 'segmentations', image_paths[i, 1].replace('.jpg', '.png'))
    rawSeg = sio.imread(seg_path)

    bbox = bboxes[i, 1:]
    w = bbox[2]
    h = bbox[3]
    x1 = int(min(max(bbox[0] - w * 0.1, 0), width))
    y1 = int(min(max(bbox[1] - h * 0.1, 0), height))
    x2 = int(min(max(bbox[0] + w * 1.1, 0), width))
    y2 = int(min(max(bbox[1] + h * 1.1, 0), height))
    cropImg = rawImg[y1:y2, x1:x2]
    cropSeg = rawSeg[y1:y2, x1:x2]

    sio.imsave(dst_path, cropImg, quality=100)
    sio.imsave(dst_path.replace('.jpg', '.png'), cropSeg)
