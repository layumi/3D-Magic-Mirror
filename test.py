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


# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import CUBDataset
from datasets.market import MarketDataset
from datasets.atr import ATRDataset
from utils import camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, compute_gradient_penalty_list, Timer
from models.model import VGG19, CameraEncoder, ShapeEncoder, LightEncoder, TextureEncoder

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='./data/CUB_Data', help='path to dataset root dir')
parser.add_argument('--gan_type', default='wgan', help='wgan or lsgan')
parser.add_argument('--template_path', default='./template/sphere.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nk', type=int, default=5, help='size of kerner')
parser.add_argument('--nf', type=int, default=32, help='dim of unit channel')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='leaning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=1, type=int, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--warm_epoch', type=int, default=20, help='warm epoch')
parser.add_argument('--multigpus', action='store_true', default=False, help='whether use multiple gpus mode')
parser.add_argument('--resume', action='store_true', default=False, help='whether resume ckpt')
parser.add_argument('--makeup', action='store_true', default=False, help='whether makeup texture')
parser.add_argument('--beta', action='store_true', default=False, help='using beta distribution instead of uniform.')
parser.add_argument('--lambda_gan', type=float, default=0.0001, help='parameter')
parser.add_argument('--lambda_reg', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_data', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_ic', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_lc', type=float, default=0.001, help='parameter')
parser.add_argument('--image_weight', type=float, default=0.1, help='parameter')
parser.add_argument('--reg', type=float, default=0.0, help='parameter')
parser.add_argument('--threshold', type=float, default=0.36, help='parameter')
parser.add_argument('--azi_scope', type=float, default=360, help='parameter')
parser.add_argument('--elev_range', type=str, default="0~30", help='elevation for mkt -15 ~ 15')
parser.add_argument('--dist_range', type=str, default="2~6", help='~ separated list of classes for the lsun data set')

opt = parser.parse_args()
opt.outf = './log/'+ opt.name
print(opt)

if not os.path.isdir(opt.outf):
    os.mkdir(opt.outf)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

### load option
with open('log/%s/opts.yaml'%opt.name,'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

opt.name = config['name']
opt.dataroot = config['dataroot']
opt.gan_type = config['gan_type']
opt.template_path = config['template_path']
opt.category = config['category']
opt.workers = config['workers']
opt.batchSize = config['batchSize']
opt.imageSize = config['imageSize']
opt.nk = config['nk']
opt.nf = config['nf']
opt.niter = config['niter']
opt.makeup = config['makeup']
opt.azi_scope = config['azi_scope']
opt.elev_range = config['elev_range']
opt.dist_range = config['dist_range']
if 'threshold' in config:
    opt.threshold = config['threshold']

if torch.cuda.is_available():
    cudnn.benchmark = True

if "MKT" in opt.name:
    train_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=True, threshold=opt.threshold)
    test_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=False, threshold=opt.threshold)
    print('Market-1501')
elif "ATR" in opt.name:
    train_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=True)
    test_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=False)
    print('ATR-human')
else:
    train_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=True)
    test_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=False)
    print('CUB')

torch.set_num_threads(int(opt.workers)*2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True, pin_memory=True, num_workers=int(opt.workers),
                                         prefetch_factor=2, persistent_workers=True) # for pytorch>1.6.0
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, pin_memory=True,
                                         num_workers=int(opt.workers), prefetch_factor=2)



if __name__ == '__main__':
    # differentiable renderer
    template_file = kal.io.obj.import_mesh(opt.template_path, with_materials=True)
    print('Vertices Number:', template_file.vertices.shape[0]) #642
    print('Faces Number:', template_file.faces.shape[0])  #1280
    diffRender = DiffRender(mesh=template_file, image_size=opt.imageSize, image_weight=opt.image_weight)

    # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
    netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                            azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                            nc=4, nk=opt.nk, nf=opt.nf, makeup=opt.makeup)

    if opt.multigpus:
        netE = torch.nn.DataParallel(netE)
    netE = netE.cuda()

    # netL: for Landmark Consistency
    # print(diffRender.num_faces) # 1280
    netL = Landmark_Consistency(num_landmarks=diffRender.num_faces, dim_feat=256, num_samples=64)
    if opt.multigpus:
        netL = torch.nn.DataParallel(netL)
    netL = netL.cuda()

    # netD: Discriminator rgb+seg
    if opt.gan_type == 'wgan':
        netD = Discriminator(nc=4, nf=32)
    elif opt.gan_type == 'lsgan':
        netD = MS_Discriminator(nc=4, nf=32)
    else:
        print('unknow gan type. Only lsgan or wgan is accepted.')

    if opt.multigpus:
        netD = torch.nn.DataParallel(netD)
    netD = netD.cuda()

    # restore from latest_ckpt.path
    start_iter = 0
    start_epoch = 0
    resume_path = os.path.join(opt.outf, 'ckpts/latest_ckpt.pth')
    if os.path.exists(resume_path):
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']
        start_iter = 0
        netD.load_state_dict(checkpoint['netD'])
        netE.load_state_dict(checkpoint['netE'])

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))

    ori_dir = os.path.join(opt.outf, 'fid/ori')
    rec_dir = os.path.join(opt.outf, 'fid/rec')
    inter_dir = os.path.join(opt.outf, 'fid/inter')
    inter90_dir = os.path.join(opt.outf, 'fid/inter90')
    ckpt_dir = os.path.join(opt.outf, 'ckpts')
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(inter90_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    summary_writer = SummaryWriter(os.path.join(opt.outf + "/logs"))
    output_txt = './log/%s/result.txt'%opt.name

    netE.eval()
    dists = torch.tensor([]).cuda()
    elevations = torch.tensor([]).cuda()
    for i, data in tqdm.tqdm(enumerate(test_dataloader)):
        Xa = Variable(data['data']['images']).cuda()
        paths = data['data']['path']
        with torch.no_grad():
            Ae = netE(Xa)
            Xer, Ae = diffRender.render(**Ae)

            #print('max: {}\nmin: {}\navg: {}'.format(torch.max(Ae['distances']), torch.min(Ae['distances']), torch.mean(Ae['distances'])))
            dists = torch.cat((dists, Ae['distances']))
            elevations = torch.cat((elevations, Ae['elevations']))

            Ai = deep_copy(Ae)
            Ai2 = deep_copy(Ae)
            Ai90 = deep_copy(Ae)
            Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
            Ai2['azimuths'] = Ai['azimuths'] + 90.0 # -90, 270
            Ai2['azimuths'][Ai2['azimuths']>180] -= 360.0 # -180, 180

            Ai90['azimuths'] += 90.0

            Xir, Ai = diffRender.render(**Ai)
            Xir2, Ai2 = diffRender.render(**Ai2)
            Xir90, Ai90 = diffRender.render(**Ai90)
                    
            for i in range(len(paths)):
                path = paths[i]
                image_name = os.path.basename(path)
                rec_path = os.path.join(rec_dir, image_name)
                output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
                output_Xer.save(rec_path, 'JPEG', quality=100)

                inter_path = os.path.join(inter_dir, image_name)
                output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
                output_Xir.save(inter_path, 'JPEG', quality=100)

                inter_path2 = os.path.join(inter_dir, '2+'+image_name)
                output_Xir2 = to_pil_image(Xir2[i, :3].detach().cpu())
                output_Xir2.save(inter_path2, 'JPEG', quality=100)

                inter90_path = os.path.join(inter90_dir, image_name)
                output_Xir90 = to_pil_image(Xir90[i, :3].detach().cpu())
                output_Xir90.save(inter90_path, 'JPEG', quality=100)

                ori_path = os.path.join(ori_dir, image_name)
                output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
                output_Xa.save(ori_path, 'JPEG', quality=100)
    print('Distance max: {}\nmin: {}\navg: {}'.format(torch.max(dists), torch.min(dists), torch.mean(dists)))
    print('Elevations max: {}\nmin: {}\navg: {}'.format(torch.max(elevations), torch.min(elevations), torch.mean(elevations)))
    fid_90 = calculate_fid_given_paths([ori_dir, inter90_dir], 32, True)
    print('Test side fid: %0.2f' % fid_90 ) 
    fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 32, True)
    print('Test recon fid: %0.2f' % fid_recon ) 
    fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 32, True)
    print('Test rotation fid: %0.2f' %  fid_inter)
