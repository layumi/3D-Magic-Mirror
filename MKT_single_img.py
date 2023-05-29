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
from PIL import Image, ImageFilter, ImageOps
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices

# draw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import CUBDataset
from datasets.market import MarketDataset
from datasets.atr import ATRDataset
from smr_utils import fliplr, mask, camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, compute_gradient_penalty_list, Timer
from network.model_res import VGG19, CameraEncoder, ShapeEncoder, LightEncoder, TextureEncoder


def save_img(output_name):
    output, name = output_name
    output.save(name, 'JPEG', quality=100)
    return


parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline-MKT', help='folder to output images and model checkpoints')
parser.add_argument('--configs_yml', default='configs/image.yml', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='../Market/hq/seg_hmr', help='path to dataset root dir')
parser.add_argument('--ratio', type=int, default=2, help='height/width')
parser.add_argument('--gan_type', default='wgan', help='wgan or lsgan')
parser.add_argument('--template_path', default='./template/ellipsoid.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--pretrains', type=str, default='none', help='pretrain shape encoder')
parser.add_argument('--pretrainc', type=str, default='none', help='pretrain camera encoder')
parser.add_argument('--pretraint', type=str, default='none', help='pretrain texture encoder')
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
opt.batchSize = config['batchSize']
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
opt.pretrains = config['pretrains']
opt.pretrainc = config['pretrainc']
opt.pretraint = config['pretraint']
opt.norm = config['norm']
opt.threshold = config['threshold']
opt.droprate = config['droprate']
opt.ratio = config['ratio']

print(opt)

if torch.cuda.is_available():
    cudnn.benchmark = True

if "MKT" in opt.name:
    train_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=True, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr)
    test_dataset = MarketDataset(opt.dataroot, opt.imageSize, train=False, threshold=opt.threshold, bg = opt.bg, hmr = opt.hmr)
    print('Market-1501')
    ratio = 2
elif "ATR" in opt.name:
    train_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = ATRDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg)
    print('ATR-human')
    ratio = 1
else:
    train_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=True, bg = opt.bg)
    test_dataset = CUBDataset(opt.dataroot, opt.imageSize, train=False, bg = opt.bg)
    print('CUB')
    ratio = 1

torch.set_num_threads(int(opt.workers)*2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                          shuffle=True, drop_last=True, pin_memory=True, num_workers=int(opt.workers),
                                          prefetch_factor=2, persistent_workers=True) # for pytorch>1.6.0
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=True, pin_memory=True,
                                         num_workers=int(opt.workers), prefetch_factor=2)



if __name__ == '__main__':
    # differentiable renderer
    template_file = kal.io.obj.import_mesh(opt.template_path, with_materials=True)


    # load updated template
    resume_path = os.path.join(opt.outf, 'ckpts/latest_ckpt.pth')
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        epoch = checkpoint['epoch']
        
    diffRender = DiffRender(mesh_name=opt.template_path, image_size=opt.imageSize, ratio = opt.ratio, image_weight=opt.image_weight)
    #latest_template_file = kal.io.obj.import_mesh(opt.outf + '/epoch_{:03d}_template.obj'.format(epoch), with_materials=True)
    #print('Loading template as epoch_{:03d}_template.obj'.format(epoch))
    latest_template_file = kal.io.obj.import_mesh(opt.outf + '/ckpts/best_mesh.obj', with_materials=True)
    diffRender.vertices_init = latest_template_file.vertices

    print('Vertices Number:', template_file.vertices.shape[0]) #642
    print('Faces Number:', template_file.faces.shape[0])  #1280

    # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
    netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                            azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                            nc=4, nk=opt.nk, nf=opt.nf, ratio=opt.ratio, makeup=opt.makeup, bg = opt.bg, 
                            pretrains = opt.pretrains, pretrainc = opt.pretrainc, pretraint = opt.pretraint,
                            droprate = opt.droprate, romp = opt.romp, 
                            coordconv = opt.coordconv, norm = opt.norm, lpl = diffRender.vertices_laplacian_matrix) # height = 2 * width

    if opt.multigpus:
        netE = torch.nn.DataParallel(netE)
    netE = netE.cuda()

    os.makedirs('demo_single', exist_ok=True)
    # restore from latest_ckpt.path
    # start_iter = 0
    # start_epoch = 0
    resume_path = os.path.join(opt.outf, 'ckpts/latest_ckpt.pth')
    if os.path.exists(resume_path):
        # Map model to be loaded to specified single gpu.
        # checkpoint has been loaded
        # start_epoch = checkpoint['epoch']
        # start_iter = 0
        #netD.load_state_dict(checkpoint['netD'])
        netE.load_state_dict(checkpoint['netE'], strict=False)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))

    netE = netE.eval()


    dists = torch.tensor([]).cuda()
    azimuths = torch.tensor([]).cuda()
    biases = torch.tensor([]).cuda()
    elevations = torch.tensor([]).cuda()
    x = torch.tensor([]).cuda()
    y = torch.tensor([]).cuda()
    z = torch.tensor([]).cuda()
    filename = []
    X_all = []
    path_all = []

    seg_path = '../Market/hq/seg_hmr/query/0005/0005_c1s1_001351_00.jpg_0.31.png'
    img_path = seg_path.replace('seg_hmr', 'pytorch')[:-9] + '.png'
    #img_path = '../Market/hq/pytorch/query/0345/0345_c6s1_079326_00.jpg.png'
    #seg_path = '../Market/hq/seg_hmr/query/1064/1064_c5s2_143149_00.jpg_0.27.png'
    #img_path = '../Market/hq/pytorch/query/1064/1064_c5s2_143149_00.jpg.png'
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_path).convert('L').point(lambda p: p > 0 and 255)
    target_width = opt.imageSize
    img = img.resize((int(target_width), int(target_width*2)))
    seg = seg.resize((int(target_width), int(target_width*2)), Image.NEAREST)
    img = torchvision.transforms.functional.to_tensor(img)
    seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]
    if opt.bg:
        rgb = img
    else:
        rgb = img * seg + torch.ones_like(img) * (1 - seg)

    rgbs = torch.cat([rgb, seg], dim=0)
    Xa = Variable(torch.unsqueeze(rgbs, 0) ).cuda()
    noise = {}
    noise[0] = torch.zeros(Xa[0,3,:,:].shape).cuda()
    noise[1] = (torch.rand(Xa[0,3,:,:].shape)<0.1).cuda()
    with torch.no_grad():
        for i in range(3):
            Xa_clone = Xa.clone()
            if i==1:
                Xa_clone[0,3,:,:] += noise[i]
            elif i==2:
                blur= torchvision.transforms.GaussianBlur(7, sigma=3)
                Xa_clone[0,3,:,:] = blur(Xa_clone[0,3,:,:].unsqueeze(0)).squeeze()
            print(Xa_clone.shape)
            M = Xa_clone[0,3,:,:].unsqueeze(0).repeat(3,1,1)
            Ae = netE(Xa_clone)
            Xer, Ae = diffRender.render(**Ae)
            
            azimuths = torch.cat((azimuths, Ae['azimuths']))
            biases = torch.cat((biases, Ae['biases']))
            dists = torch.cat((dists, Ae['distances']))
            elevations = torch.cat((elevations, Ae['elevations']))
            x = torch.cat((x, Ae['delta_vertices'][:,:,0]))
            y = torch.cat((y, Ae['delta_vertices'][:,:,1]))
            z = torch.cat((z, Ae['delta_vertices'][:,:,2]))

            Ai = deep_copy(Ae)
            Ai2 = deep_copy(Ae)
            Ae90 = deep_copy(Ae)
            Ai['azimuths'] = Ai['azimuths'] + 45
            Ai['azimuths'][Ai['azimuths']>180] -= 360.0
            #- torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
            Ai2['azimuths'] = Ai['azimuths'] + 90.0 # -90, 270
            Ai2['azimuths'][Ai2['azimuths']>180] -= 360.0 # -180, 180

            Ae90['azimuths'] += 90.0

            Xir, Ai = diffRender.render(**Ai)
            Xir2, Ai2 = diffRender.render(**Ai2)
            Xer90, Ae90 = diffRender.render(**Ae90)
            ###
            #Ae90_recon = netE(Xer90) 
            #print(Ae90_recon['azimuths'])
            #break
            Xa_clone = mask(Xa_clone) # remove bg
                   
            path = img_path
            filename.append(path)
            image_name = os.path.basename(path) + '.jpg'
            rec_path = 'demo_single/%d_rec_'%i+ image_name
            output_Xer = to_pil_image(Xer[0, :3].detach().cpu())
            inter_path = 'demo_single/%d_inter_'%i+ image_name
            output_Xir = to_pil_image(Xir[0, :3].detach().cpu())
            ori_path = 'demo_single/%d_ori_'%i+ image_name
            output_Xa = to_pil_image(Xa[0, :3].detach().cpu())
            msk_path = 'demo_single/%d_msk_'%i+ image_name
            output_Mask = to_pil_image(M.detach().cpu())

            all_path = 'demo_single/%d_concat_'%i+ image_name
            Xall = torch.concat( (Xa[0, :3], M, Xer[0, :3], Xir[0, :3]), dim=2 )
            output_All = to_pil_image(Xall.detach().cpu())

            # Single Output
            #X_all.extend([output_Xer, output_Xir, output_Xa, output_Mask, output_All])
            #path_all.extend([rec_path, inter_path, ori_path, msk_path, all_path])
            
            X_all.extend([output_All])
            path_all.extend([all_path])

            # gif
            print('===========Saving Gif-Azi===========')
            rotate_path = 'demo_single/%d_'%i+ image_name + '_rotation.gif'
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(range(0, int(opt.azi_scope), 10))) # 0, 360
            A_tmp = deep_copy(Ae)
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')
            for delta_azimuth in loop:
                A_tmp['azimuths'] = Ae['azimuths'] -  delta_azimuth
                predictions, _ = diffRender.render(**A_tmp)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = torch.concat( (Xa[0, :3], M, image), dim=2 )
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()


    with Pool(4) as p:
        p.map(save_img, zip(X_all, path_all) )


    azimuths_result = 'Azimuths max: {}\tmin: {}\tavg: {}'.format(torch.max(azimuths), torch.min(azimuths), torch.mean(azimuths))
    biases_result = 'Biases-X max: {}\tmin: {}\tavg: {}\n'.format(torch.max(biases[:,0]), torch.min(biases[:,0]), torch.mean(biases[:,0]))
    biases_result += 'Biases-Y max: {}\tmin: {}\tavg: {}'.format(torch.max(biases[:,1]), torch.min(biases[:,1]), torch.mean(biases[:,1]))
    dist_result = 'Distances max: {}\tmin: {}\tavg: {}'.format(torch.max(dists), torch.min(dists), torch.mean(dists))
    elev_result = 'Elevations max: {}\tmin: {}\tavg: {}'.format(torch.max(elevations), torch.min(elevations), torch.mean(elevations))
    xyz_result = 'X max: {}\t min: {}\t avg: {}'.format(torch.max(x, dim=1)[0], torch.min(x, dim = 1)[0], torch.mean(x, dim=1))


    fig = plt.figure()
    ax0 = fig.add_subplot(231, title="Azimuths")
    ax1 = fig.add_subplot(232, title="X")
    ax2 = fig.add_subplot(233, title="Y")
    ax3 = fig.add_subplot(234, title="Z")
    ax4 = fig.add_subplot(235, title="Elevations")
    ax5 = fig.add_subplot(236, title="Shape Biase Max")
    ax0.hist( azimuths.cpu().numpy(), 36, density=True, facecolor='g', alpha=0.75)
    ax1.hist( x.cpu().numpy(), 5, density=True, facecolor='g', alpha=0.75)
    ax2.hist( y.cpu().numpy(), 5, density=True, facecolor='g', alpha=0.75)
    ax3.hist( z.cpu().numpy(), 5, density=True, facecolor='g', alpha=0.75)
    ax4.hist( elevations.cpu().numpy(), 36, density=True, facecolor='g', alpha=0.75)
    ax5.hist( x.cpu().numpy(), 36, density=True, facecolor='g', alpha=0.75)
    fig.savefig("demo_single/hist.png")

    print(azimuths_result)
    print(biases_result)
    print(dist_result)
    print(elev_result)
    print(xyz_result)
    

