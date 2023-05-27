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
from multiprocessing import Pool
# import torch related
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
parser.add_argument('--ellipsoid', type=float, default = 2, help='init sphere to ellipsoid' )
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--pretrainc', type=str, default='none', help='pretrain shape encoder')
parser.add_argument('--pretrains', type=str, default='hr18sv2', help='pretrain shape encoder')
parser.add_argument('--pretraint', type=str, default='res34', help='pretrain shape encoder')
parser.add_argument('--norm', type=str, default='bn', help='norm function')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
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
parser.add_argument('--bg', action='store_true', default=False, help='use background')
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
parser.add_argument('--threshold', type=float, default=0.09, help='parameter')
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
    opt.manualSeed = random.randint(1, 10000)
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
opt.pretrainc = config['pretrainc']
opt.pretrains = config['pretrains']
opt.pretraint = config['pretraint']
opt.norm = config['norm']
opt.threshold = config['threshold']
opt.droprate = config['droprate']
opt.ratio = config['ratio']
opt.ellipsoid = config['ellipsoid']

print(opt)


if torch.cuda.is_available():
    cudnn.benchmark = True

if "MKT" in opt.name:
    selected_index = np.arange(1, 3368, int(3368//opt.batchSize)) 
    print(selected_index) 
    # more challenge cases
    selected_index[0] = 2328
    selected_index[1] = 2614
    selected_index[2] = 2114
    selected_index[3] = 1897
    selected_index[4] = 2852
    selected_index[5] = 1236
    selected_index[6] = 2333
    selected_index[7] = 388
    selected_index[8] = 1300
    selected_index[9] = 83
    selected_index[10] = 2293
    selected_index[11] = 503
    selected_index[12] = 1586
    selected_index[13] = 3100
    selected_index[14] = 1121
    selected_index[15] = 1082
    selected_index[16] = 1997
    selected_index[17] = 2061
    selected_index[18] = 77
    selected_index[19] = 2238 
    selected_index[20] = 1236
    selected_index[21] = 1268
    selected_index[22] = 2232
    selected_index[23] = 863
    selected_index[24] = 3258
    selected_index[25] = 3213
    selected_index[26] = 555
    selected_index[27] = 2582
    selected_index[28] = 867
    selected_index[29] = 1290
    selected_index[30] = 2333
    selected_index[31] = 3155
    #print(selected_index) 
    train_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=True, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr)
    test_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=False, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr, selected_index = selected_index)
    print('Market-1501: %d'% len(test_dataset))
    ratio = 2
elif "ATR" in opt.name:
    selected_index = np.arange(70, 16000, 16000//opt.batchSize) 
    print(selected_index)
    selected_index[0] = 13596
    selected_index[4] = 11001
    selected_index[11] = 8004
    selected_index[13] = 1005
    selected_index[14] = 14080
    selected_index[15] = 13580
    train_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg,  selected_index = selected_index)
    print('ATR-human: %d'% len(test_dataset))
    ratio = 1
else:
    selected_index = np.arange(0, 5748, 5748//opt.batchSize)
    print(selected_index)
    train_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg,  selected_index = selected_index)
    print('CUB:%d'%len(test_dataset))
    ratio = 1

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
                            pretrainc = opt.pretrainc, pretrains = opt.pretrains, pretraint = opt.pretraint,
                            droprate = opt.droprate, romp = opt.romp, 
                            coordconv = opt.coordconv, norm = opt.norm, lpl = diffRender.vertices_laplacian_matrix) # height = 2 * width

    if opt.multigpus:
        netE = torch.nn.DataParallel(netE)
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

    for i, data in tqdm.tqdm(enumerate(test_dataloader)):
    #for i, data in tqdm.tqdm(enumerate(train_dataloader)):
        Xa = Variable(data['data']['images']).cuda()
        paths = data['data']['path']
        print(paths)
        # Xa = fliplr(Xa)
        with torch.no_grad():
            Ae = netE(Xa)
            Xer, Ae = diffRender.render(**Ae)
            
            azimuths = torch.cat((azimuths, Ae['azimuths']))
            biases = torch.cat((biases, Ae['biases']))
            dists = torch.cat((dists, Ae['distances']))
            elevations = torch.cat((elevations, Ae['elevations']))
            xyz_min = torch.cat((xyz_min, torch.min(Ae['delta_vertices'], dim=1)[0]))
            xyz_max = torch.cat((xyz_max, torch.max(Ae['delta_vertices'], dim=1)[0]))
            xyz_mean = torch.cat((xyz_mean, torch.mean(torch.abs(Ae['delta_vertices']), dim=1)))

            Ai = deep_copy(Ae)
            Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()

            Xir, Ai = diffRender.render(**Ai)
            textures = Ae['textures'] 
            Xa = (Xa * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            Xer = (Xer * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            Xir = (Xir * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

            Xa = torch.tensor(Xa, dtype=torch.float32) / 255.0
            Xa = Xa.permute(0, 3, 1, 2)
            Xer = torch.tensor(Xer, dtype=torch.float32) / 255.0
            Xer = Xer.permute(0, 3, 1, 2)
            Xir = torch.tensor(Xir, dtype=torch.float32) / 255.0
            Xir = Xir.permute(0, 3, 1, 2)


            #############
            opt.outf = './rainbow/'
            ########
            vutils.save_image(Xa[:, :3],
                    '%s/current_Xa.png' % (opt.outf), normalize=True, nrow = nrow)

            vutils.save_image(Xer[:, :3].detach(),
                    '%s/current_Xer.png' % (opt.outf), normalize=True, nrow = nrow)

            vutils.save_image(Xir[:, :3].detach(),
                    '%s/current_Xir.png' % (opt.outf), normalize=True, nrow = nrow)

            vutils.save_image(textures.detach(),
                    '%s/current_textures.png' % (opt.outf), normalize=True, nrow = nrow)

            #vutils.save_image(Ea.detach(),
            #        '%s/current_edge.png' % (opt.outf), normalize=True)

            Ae = deep_copy(Ae, detach=True)
            vertices = Ae['vertices']
            faces = diffRender.faces
            uvs = diffRender.uvs
            textures = Ae['textures']
            azimuths = Ae['azimuths']
            elevations = Ae['elevations']
            distances = Ae['distances']
            lights = Ae['lights']

            texure_maps = to_pil_image(textures[0].detach().cpu())
            texure_maps.save('%s/current_mesh_recon.png' % (opt.outf), 'PNG')
            texure_maps.save('%s/epoch_%03d_mesh_recon.png' % (opt.outf, epoch), 'PNG')

            save_mesh('%s/current_mesh_recon.obj' % opt.outf, vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)
            save_mesh('%s/epoch_%03d_template.obj' % (opt.outf, epoch), netE.vertices_init[0].clone().detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)

            print('===========Saving Rainbow===========')
            rotate_path = os.path.join(opt.outf, 'current_rainbow.gif')
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(range(0, int(opt.azi_scope), 10))) # 0, 360
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')
            A_tmp = deep_copy(Ae, detach=True, index = [0,1,2,3,4,5,6])
            batch_tmp = 7
            predictions_all = torch.ones((8, 8, 3, opt.imageSize * opt.ratio, opt.imageSize))
            predictions_all[0, 1:, :, :, : ] = Xa[0:7, :3]
            for delta_azimuth in loop:
                for i in range(batch_tmp):
                    A_tmp['azimuths'] =  - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(batch_tmp).cuda()
                    A_tmp['textures']  = Ae['textures'][i].unsqueeze(0).repeat(batch_tmp, 1, 1, 1) 
                    predictions, _ = diffRender.render(**A_tmp)
                    predictions = torch.cat( (Xa[i, :3].unsqueeze(0), predictions[:, :3].cpu()), dim=0)
                    predictions_all[i+1, :, :, :, : ] = predictions
                   
                predictions_save = predictions_all.view(-1, 3, opt.imageSize * opt.ratio, opt.imageSize)
                predictions_save = torchvision.transforms.functional.pad(predictions_save, padding=2)
                image = vutils.make_grid(predictions_save, nrow=8, padding = 20, pad_value = 1.0)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()


            print('===========Saving Gif-Azi===========')
            rotate_path = os.path.join(opt.outf, 'current_rotation.gif')
            writer = imageio.get_writer(rotate_path, mode='I')
            #loop = tqdm.tqdm(list(range(-int(opt.azi_scope/2), int(opt.azi_scope/2), 10))) # -180, 180
            loop = tqdm.tqdm(list(range(0, int(opt.azi_scope), 10))) # 0, 360
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')
            A_tmp = deep_copy(Ae, detach=True)
            for delta_azimuth in loop:
                # start from recon
                A_tmp['azimuths'] = Ae['azimuths'] - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(opt.batchSize).cuda()
                #A_tmp['azimuths'] = - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(opt.batchSize).cuda()
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions, nrow=nrow)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()

            print('===========Saving Gif-Y===========')
            rotate_path = os.path.join(opt.outf, 'current_rotation_ele.gif' )
            writer = imageio.get_writer(rotate_path, mode='I')
            elev_range = opt.elev_range.split('~')
            elev_min = int(elev_range[0])
            elev_max = int(elev_range[1])
            loop = tqdm.tqdm(list(range(elev_min, elev_max, 5))) # 15~-45
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_ele)')
            A_tmp = deep_copy(Ae, detach=True)
            for delta_elevation in loop:
                A_tmp['elevations'] = - torch.tensor([delta_elevation], dtype=torch.float32).repeat(opt.batchSize).cuda()
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions, nrow=nrow)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()

            print('===========Saving Gif-Dist===========')
            rotate_path = os.path.join(opt.outf, 'current_rotation_dist.gif')
            writer = imageio.get_writer(rotate_path, mode='I')
            dist_range = opt.dist_range.split('~')
            dist_min = int(dist_range[0])
            dist_max = int(dist_range[1])
            loop = tqdm.tqdm(list(np.arange(dist_min, dist_max+1, 0.5))) # 1, 7
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_dist)')
            A_tmp = deep_copy(Ae, detach=True)
            for delta_dist in loop:
                A_tmp['distances'] = - torch.tensor([delta_dist], dtype=torch.float32).repeat(opt.batchSize).cuda()
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions, nrow=nrow)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)

            writer.close()

            print('===========Saving Gif-XY===========')
            rotate_path = os.path.join(opt.outf, 'current_rotation_XY.gif')
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(np.arange(-0.5, 0.6, 0.1))) # 1, 7
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_biases)')
            A_tmp = deep_copy(Ae, detach=True)
            for delta_biases in loop:
                A_tmp['azimuths'] = torch.tensor([0], dtype=torch.float32).repeat(opt.batchSize).cuda()
                A_tmp['biases'] = torch.tensor([delta_biases, 0], dtype=torch.float32).unsqueeze(0).repeat(opt.batchSize,1).cuda()
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions, nrow=nrow)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            for delta_biases in loop:
                A_tmp['azimuths'] = torch.tensor([0], dtype=torch.float32).repeat(opt.batchSize).cuda()
                A_tmp['biases'] = torch.tensor([0, delta_biases], dtype=torch.float32).unsqueeze(0).repeat(opt.batchSize,1).cuda()
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions, nrow=nrow)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)

            writer.close()
            break
