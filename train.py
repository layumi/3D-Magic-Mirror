import argparse
import os
import random
import math
import tqdm
import shutil
import imageio
import numpy as np
import trimesh
import yaml

# import torch related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from networks import MS_Discriminator, Discriminator, DiffRender, Landmark_Consistency, AttributeEncoder, weights_init, deep_copy
# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices

from trainer import trainer
# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import CUBDataset
from datasets.market import MarketDataset

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--configs_yml', default='configs/image.yml', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='./data/CUB_Data', help='path to dataset root dir')
parser.add_argument('--ratio', type=int, default=1, help='height/width')
parser.add_argument('--gan_type', default='wgan', help='wgan or lsgan')
parser.add_argument('--template_path', default='./template/sphere.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--pretrain', type=str, default='hr18sv2', help='pretrain shape encoder. default is hr18sv2 or hr18 or none or hr18sv1')
parser.add_argument('--norm', type=str, default='bn', help='norm function')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nk', type=int, default=5, help='size of kerner')
parser.add_argument('--nf', type=int, default=32, help='dim of unit channel')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='leaning rate, default=0.0001')
parser.add_argument('--clip', type=float, default=0.05, help='the clip for template update.')
parser.add_argument('--azim', type=float, default=1.0, help='recon weight for azim. default=1.0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--wd', type=float, default=0, help='weight decay for adam. default=0')
parser.add_argument('--droprate', type=float, default=0.2, help='dropout in encoders. default=0.2')
parser.add_argument('--cuda', default=1, type=int, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--warm_epoch', type=int, default=40, help='warm epoch')
parser.add_argument('--multigpus', action='store_true', default=False, help='whether use multiple gpus mode')
parser.add_argument('--resume', action='store_true', default=False, help='whether resume ckpt')
parser.add_argument('--chamfer', action='store_true', default=False, help='use chamfer loss for vertices')
parser.add_argument('--amsgrad', action='store_true', default=False, help='use amsgrad')
parser.add_argument('--bg', action='store_true', default=False, help='use background')
parser.add_argument('--nolpl', action='store_true', default=False, help='ablation study for no template in camera and shape encoder')
parser.add_argument('--white', action='store_true', default=True, help='use normalized template')
parser.add_argument('--smooth', type=float, default=0.5, help='using smooth template.')
parser.add_argument('--makeup', type=int, default=0, help='whether makeup texture 0:nomakeup 1:in 2:bn 3:ln 4.none')
parser.add_argument('--beta', type=float, default=0, help='using beta distribution instead of uniform.')
parser.add_argument('--hard', action='store_true', default=False, help='using Xer90 instead of Xer.')
parser.add_argument('--adamw', action='store_true', default=False, help='using adamw.')
parser.add_argument('--L1', action='store_true', default=False, help='using L1 for ic loss.')
parser.add_argument('--flipL1', action='store_true', default=False, help='using flipL1 for flipz loss.')
parser.add_argument('--coordconv', action='store_false', default=True, help='using coordconv for texture mapping.')
parser.add_argument('--unmask', type=int, default=0, help='0 remove background, 1 for rgb only, 2 for four channel')
parser.add_argument('--romp', action='store_true', default=False, help='using romp.')
parser.add_argument('--swa', action='store_true', default=True, help='using swa.')
parser.add_argument('--em', type=float, default=1.0, help='update template')
parser.add_argument('--em_gap', type=int, default=1, help='update template evey xx epoch ')
parser.add_argument('--swa_start', type=int, default=400, help='switch to swa at epoch swa_start')
parser.add_argument('--swa_interval', type=int, default=1, help='averge model every interval epoch')
parser.add_argument('--update_shape', type=int, default=1, help='train shape every XX iteration')
parser.add_argument('--update_bn', type=int, default=1, help='update model after template update')
parser.add_argument('--swa_lr', type=float, default=0.0003, help='swa learning rate')
parser.add_argument('--lambda_gan', type=float, default=0.0001, help='parameter')
parser.add_argument('--ganw', type=float, default=1, help='parameter for Xir. Since it is hard.')
parser.add_argument('--lambda_edge', type=float, default=0.001, help='parameter to prevent long edge')
parser.add_argument('--lambda_depth', type=float, default=0.001, help='parameter to prevent long z predictions, especially on the edge.')
parser.add_argument('--lambda_deform', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_flipz', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_data', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_ic', type=float, default=1, help='parameter')
parser.add_argument('--lambda_reg', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_lpl', type=float, default=0.1, help='parameter for laplacian loss')
parser.add_argument('--lambda_flat', type=float, default=0.001, help='parameter for flatten loss')
parser.add_argument('--gamma', type=float, default=0.01, help='parameter')
parser.add_argument('--dis1', type=float, default=0, help='parameter')
parser.add_argument('--dis2', type=float, default=0, help='parameter')
parser.add_argument('--lambda_lc', type=float, default=0, help='parameter')
parser.add_argument('--image_weight', type=float, default=1, help='parameter')
parser.add_argument('--gan_reg', type=float, default=10.0, help='parameter')
parser.add_argument('--em_step', type=float, default=0.1, help='parameter')
parser.add_argument('--hmr', type=float, default=0.0, help='parameter')
parser.add_argument('--threshold', type=float, default=0.09, help='parameter')
parser.add_argument('--bias_range', type=float, default=0.3, help='parameter bias range')
parser.add_argument('--azi_scope', type=float, default=360, help='parameter')
parser.add_argument('--elev_range', type=str, default="0~30", help='~ elevantion')
parser.add_argument('--hard_range', type=int, default=0, help='~ range from x to 180-x. x<90')
parser.add_argument('--dist_range', type=str, default="2~7", help='~ separated list of classes for the lsun data set')

opt = parser.parse_args()
opt.outf = './log/'+ opt.name
print(opt)
os.makedirs('./log', exist_ok=True)
if not os.path.isdir(opt.outf):
    os.mkdir(opt.outf)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

### save option
with open('log/%s/opts.yaml'%opt.name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

if torch.cuda.is_available():
    cudnn.benchmark = True

train_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
test_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg)

torch.set_num_threads(int(opt.workers)*2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True, pin_memory=True, num_workers=int(opt.workers),
                                         prefetch_factor=2, persistent_workers=True) # for pytorch>1.6.0
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, pin_memory=True, 
                                         num_workers=int(opt.workers), prefetch_factor=2)


if __name__ == '__main__':
    trainer(opt, train_dataloader, test_dataloader)
