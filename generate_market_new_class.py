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
import scipy.io
from multiprocessing import Pool
# import torch related
from scipy.ndimage import laplace
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from networks import MS_Discriminator, Discriminator, DiffRender, Landmark_Consistency, AttributeEncoder, weights_init, deep_copy
# import kaolin related
import kaolin as kal
from kaolin.metrics.render import mask_iou
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices
from PIL import Image
from pytorch_msssim import ssim
#from skimage.metrics import structural_similarity as ssim
# draw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import CUBDataset
from datasets.market import MarketDataset
from datasets.atr import ATRDataset
from smr_utils import save_mesh, fliplr, mask, camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, compute_gradient_penalty_list, Timer
from network.model_res import VGG19, CameraEncoder, ShapeEncoder, LightEncoder, TextureEncoder


def save_img(output_name):
    output, name = output_name
    os.makedirs(os.path.dirname(name), exist_ok=True)
    output.save(name, 'JPEG', quality=100)
    return

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline-MKT', help='folder to output images and model checkpoints')
parser.add_argument('--configs_yml', default='configs/image.yml', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='../Market/hq/seg_hmr', help='path to dataset root dir')
parser.add_argument('--ratio', type=int, default=2, help='height/width')
parser.add_argument('--gan_type', default='wgan', help='wgan or lsgan')
parser.add_argument('--template_path', default='./template/ellipsoid.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--pretrain', type=str, default='none', help='pretrain shape encoder')
parser.add_argument('--norm', type=str, default='bn', help='norm function')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nk', type=int, default=5, help='size of kerner')
parser.add_argument('--nf', type=int, default=32, help='dim of unit channel')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='leaning rate, default=0.0001')
parser.add_argument('--clip', type=float, default=0.05, help='the clip for template update.')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--droprate', type=float, default=0.2, help='dropout in encoders. default=0.2')
parser.add_argument('--cuda', default=1, type=int, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--warm_epoch', type=int, default=20, help='warm epoch')
parser.add_argument('--multigpus', action='store_true', default=False, help='whether use multiple gpus mode')
parser.add_argument('--resume', action='store_true', default=False, help='whether resume ckpt')
parser.add_argument('--chamfer', action='store_true', default=False, help='use chamfer loss for vertices')
parser.add_argument('--bg', action='store_true', default=True, help='use background')
parser.add_argument('--nolpl', action='store_true', default=False, help='ablation study for no template in camera and shape encoder')
parser.add_argument('--white', action='store_true', default=False, help='use normalized template')
parser.add_argument('--makeup', type=int, default=0, help='whether makeup texture 0:nomakeup 1:in 2:bn 3:ln 4.none')
parser.add_argument('--beta', type=float, default=0, help='using beta distribution instead of uniform.')
parser.add_argument('--hard', action='store_true', default=False, help='using Xer90 instead of Xer.')
parser.add_argument('--L1', action='store_true', default=False, help='using L1 for ic loss.')
parser.add_argument('--flipL1', action='store_true', default=False, help='using flipL1 for flipz loss.')
parser.add_argument('--coordconv', action='store_false', default=True, help='using coordconv for texture mapping.')
parser.add_argument('--unmask', action='store_true', default=False, help='using L1 for ic loss.')
parser.add_argument('--romp', action='store_true', default=False, help='using romp.')
parser.add_argument('--swa', action='store_true', default=False, help='using swa.')
parser.add_argument('--em', type=float, default=0.0, help='update template')
parser.add_argument('--swa_start', type=int, default=400, help='switch to swa at epoch swa_start')
parser.add_argument('--update_shape', type=int, default=1, help='train shape every XX iteration')
parser.add_argument('--swa_lr', type=float, default=0.0003, help='swa learning rate')
parser.add_argument('--lambda_gan', type=float, default=0.0001, help='parameter')
parser.add_argument('--ganw', type=float, default=1, help='parameter for Xir. Since it is hard.')
parser.add_argument('--lambda_reg', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_edge', type=float, default=0.001, help='parameter')
parser.add_argument('--lambda_deform', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_flipz', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_data', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_ic', type=float, default=1, help='parameter')
parser.add_argument('--dis1', type=float, default=0, help='parameter')
parser.add_argument('--dis2', type=float, default=0, help='parameter')
parser.add_argument('--lambda_lc', type=float, default=0, help='parameter')
parser.add_argument('--image_weight', type=float, default=1, help='parameter')
parser.add_argument('--reg', type=float, default=0.0, help='parameter')
parser.add_argument('--em_step', type=float, default=0.1, help='parameter')
parser.add_argument('--hmr', type=float, default=0.0, help='parameter')
parser.add_argument('--threshold', type=float, default=0, help='parameter')
parser.add_argument('--bias_range', type=float, default=0.5, help='parameter bias range')
parser.add_argument('--azi_scope', type=float, default=360, help='parameter')
parser.add_argument('--elev_range', type=str, default="-25~25", help='~ elevantion')
parser.add_argument('--hard_range', type=int, default=0, help='~ range from x to 180-x. x<90')
parser.add_argument('--dist_range', type=str, default="2~6", help='~ separated list of classes for the lsun data set')

opt = parser.parse_args()
opt.outf = './log/'+ opt.name
if not os.path.isdir(opt.outf):
    os.mkdir(opt.outf)

if opt.manualSeed is None:
    opt.manualSeed = 666  #random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

### load option
with open('log/%s/opts.yaml'%opt.name,'r') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

opt.template_path = config['template_path']
opt.name = config['name']
opt.dataroot = config['dataroot']
opt.gan_type = config['gan_type']
opt.category = config['category']
opt.workers = config['workers']
#opt.batchSize = config['batchSize']
opt.imageSize = config['imageSize']
opt.nk = config['nk']
opt.nf = config['nf']
opt.niter = config['niter']
opt.makeup = config['makeup']
opt.azi_scope = config['azi_scope']
opt.bias_range = config['bias_range']
opt.elev_range= config['elev_range']
opt.dist_range = config['dist_range']
opt.bg = config['bg']
opt.coordconv = config['coordconv']
opt.pretrain = config['pretrain']
opt.norm = config['norm']
opt.threshold = config['threshold']
opt.droprate = config['droprate']
opt.ratio = config['ratio']

print(opt)


if torch.cuda.is_available():
    cudnn.benchmark = True

if "MKT" in opt.name:
    #print(selected_index) 
    train_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=False, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr, sub='train_all')
    test_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=False, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr)
    print('Market-1501:%d'% len(test_dataset))
    ratio = 2
elif "ATR" in opt.name:
    train_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg,  selected_index = selected_index)
    print('ATR-human: %d'% len(test_dataset))
    ratio = 1
else:
    train_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg)
    print('CUB: %d'%len(test_dataset))
    ratio = 1

torch.set_num_threads(int(opt.workers)*2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                          shuffle=False, pin_memory=True, num_workers=int(opt.workers),
                                          prefetch_factor=2, persistent_workers=True) # for pytorch>1.6.0
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, pin_memory=True,
                                         num_workers=int(opt.workers), prefetch_factor=2)



if __name__ == '__main__':
    # differentiable renderer
    template_file = kal.io.obj.import_mesh(opt.template_path, with_materials=True)


    # load updated template
    resume_path = os.path.join(opt.outf, 'ckpts/best_ckpt.pth')
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        epoch = checkpoint['epoch']
        
    diffRender = DiffRender(mesh_name=opt.template_path, image_size=opt.imageSize, ratio = opt.ratio, image_weight=opt.image_weight)
    #latest_template_file = kal.io.obj.import_mesh(opt.outf + '/epoch_{:03d}_template.obj'.format(epoch), with_materials=True)
    latest_template_file = kal.io.obj.import_mesh(opt.outf + '/ckpts/best_mesh.obj', with_materials=True)
    #print('Loading template as epoch_{:03d}_template.obj'.format(epoch))
    diffRender.vertices_init = latest_template_file.vertices

    print('Vertices Number:', template_file.vertices.shape[0]) #642
    print('Faces Number:', template_file.faces.shape[0])  #1280

    # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
    netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                            azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                            nc=4, nk=opt.nk, nf=opt.nf, ratio=opt.ratio, makeup=opt.makeup, bg = opt.bg, 
                            pretrain = opt.pretrain, droprate = opt.droprate, romp = opt.romp, 
                            coordconv = opt.coordconv, norm = opt.norm, lpl = diffRender.vertices_laplacian_matrix) # height = 2 * width

    #if opt.multigpus:
    netE = netE.cuda()

    # restore from latest_ckpt.path
    resume_path = os.path.join(opt.outf, 'ckpts/best_ckpt.pth')
    if os.path.exists(resume_path):
        # Map model to be loaded to specified single gpu.
        # checkpoint has been loaded
        # start_epoch = checkpoint['epoch']
        # start_iter = 0
        #netD.load_state_dict(checkpoint['netD'])
        netE.load_state_dict(checkpoint['netE'], strict=True)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))

    netE = netE.eval()
    #netE = torch.nn.DataParallel(netE)
    dists = torch.tensor([]).cuda()
    azimuths = torch.tensor([]).cuda()
    biases = torch.tensor([]).cuda()
    elevations = torch.tensor([]).cuda()
    xyz_min = torch.tensor([]).cuda()
    xyz_mean = torch.tensor([]).cuda()
    xyz_max = torch.tensor([]).cuda()
    filename = []
    X_all = []
    path_all = []

    if  opt.ratio == 2: 
        nrow = 8
    else: 
        nrow = 4

    #############
    opt.outf = '../Magic_Market2/'
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.outf+'/hq', exist_ok=True)
    os.makedirs(opt.outf+'/hq/pytorch', exist_ok=True)
    os.makedirs(opt.outf+'/hq/pytorch/train_all', exist_ok=True)
    os.makedirs(opt.outf+'/hq/pytorch/gallery', exist_ok=True)
    os.makedirs(opt.outf+'/hq/pytorch/query', exist_ok=True)


    mean_textures = {}
    #for i, data in tqdm.tqdm(enumerate(test_dataloader)):
    for i, data in tqdm.tqdm(enumerate(train_dataloader)):
        Xa = data['data']['images'].cuda()
        paths = data['data']['path']
        with torch.no_grad():
            Ae = netE(Xa)
            Xer, Ae = diffRender.render(**Ae)

            Ae = deep_copy(Ae, detach=True)
            textures = Ae['textures'].cpu()
            for i in range(Xa.shape[0]):
                person_id = os.path.basename(paths[i]).split('_')[0]
                if str(person_id) in mean_textures:
                    mean_textures[str(person_id)].append(textures[i])
                else:
                    mean_textures[str(person_id)] = [textures[i]]

    count = 0
    mean_tensor = torch.FloatTensor(751, 3, 256, 64).cuda()
    mean_name = []
    for person_id in mean_textures:
        mean_textures[person_id] = torch.mean( torch.stack(mean_textures[person_id]), dim=0)
        mean_tensor[count,:,:,:] = mean_textures[person_id]
        mean_name.append(person_id)
        count += 1

    for i, data in tqdm.tqdm(enumerate(train_dataloader)):
        Xa = data['data']['images'].cuda()
        paths = data['data']['path']
        #* (1 - Xa[:,3].unsqueeze(1))
        # Xa = fliplr(Xa)
        with torch.no_grad():
            Ae = netE(Xa)
            Xer, Ae = diffRender.render(**Ae)

            ########
            Ae = deep_copy(Ae, detach=True)
            vertices = Ae['vertices']
            faces = diffRender.faces
            uvs = diffRender.uvs
            textures = Ae['textures']
            azimuths = Ae['azimuths']
            elevations = Ae['elevations']
            distances = Ae['distances']
            lights = Ae['lights']

            im_list = []
            name_list = []
            print('===========Saving Gif-Azi===========')
            A_tmp = deep_copy(Ae, detach=True)
            loop = tqdm.tqdm(list([-30, -15, 15, 30])) # 30, 60 
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')

            bg = Xa[:,:3] #* (1 - Xa[:,3].unsqueeze(1))
            padding  = torch.nn.ReflectionPad2d(16)
            imresize = torchvision.transforms.Resize(size= (Xa.shape[2], Xa.shape[3]))
            bg = padding(bg)
            gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=31, sigma=2.0)
            current_batchSize = Xa.shape[0]

            rand_person_id = torch.randperm(751)[0:current_batchSize].cuda()
            for delta_azimuth in loop:
                A_tmp['azimuths'] = Ae['azimuths'] - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(current_batchSize).cuda()
                A_tmp['distances'] = Ae['distances'] - 0.5*torch.randn(current_batchSize).cuda()
                A_tmp['elevations'] = Ae['elevations'] - 0.1*torch.randn(current_batchSize).cuda()
                #A_tmp['textures'] = 0.5 * Ae['textures'] + 0.5 * mean_tensor[rand_person_id].repeat(current_batchSize,1,1,1)
                A_tmp['textures'] =  0.5 * Ae['textures'] + 0.5 * torch.index_select(mean_tensor, dim=0, index = rand_person_id)
                predictions, _ = diffRender.render(**A_tmp)
                mask = predictions[:, 3]#.unsqueeze(1) # B*C*H*W
                image = predictions[:, :3]
                for i in range(current_batchSize): 
                    single_image = image[i,:,:,:] # bg is white 1
                    single_mask  = mask[i,:,:] # bg is black 0, fg is white 1
                    blur_mask = gaussian_blur(single_mask.unsqueeze(0))
                    blur_bg = gaussian_blur(bg[i,:,:,:])
                    blur_bg = imresize(blur_bg)
                    out_image = single_image * blur_mask + blur_bg * (1-blur_mask)
                    out_image = Image.fromarray(np.uint8(out_image.cpu().numpy().transpose(1,2,0)*255))    
                    im_list.append(out_image)
                    p = paths[i]
                    old_id = os.path.basename(p).split('_')[0]
                    new_id = mean_name[rand_person_id[i]]
                    if int(old_id) < int(new_id):
                        dir_id = old_id+new_id
                    elif int(old_id) == int(new_id):
                        continue
                    else:
                        dir_id = new_id+old_id
                    outp = os.path.dirname(os.path.dirname(p)).replace('Market', 'Magic_Market')+'/'+ dir_id +'/'+os.path.basename(p)[:-8]+'%03d.jpg'%delta_azimuth
                    name_list.append(outp)
            with Pool(4) as p:
                p.map(save_img, zip(im_list, name_list))
            os.system('rsync -r ../Market/pytorch/* ../Magic_Market/hq/pytorch/')
