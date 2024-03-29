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
import copy
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
from torchvision.transforms.transforms import ColorJitter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import DBSCAN
from networks import MS_Discriminator, Discriminator, DiffRender, Landmark_Consistency, AttributeEncoder, weights_init, deep_copy
# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices

from PIL import Image
import time
from pytorch_msssim import ssim
from kaolin.metrics.render import mask_iou
from pytorch3d.loss import chamfer_distance
#from chamferdist import ChamferDistance
# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import CUBDataset
from datasets.market import MarketDataset
from datasets.atr import ATRDataset
from smr_utils import face_clocks, angle2xy, white, iou_pytorch, save_mesh, mask, ChannelShuffle, fliplr, camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, compute_gradient_penalty_list, Timer

def save_img(output_name):
    output, name = output_name
    output.save(name, 'JPEG', quality=100)
    return

def regularization(diffRender, Ae, Ai, Aire, opt):
    # lossR_reg, lossR_flip, lossR_IC = regularization(diffRender, Ae, Ai, opt)
    lossR_reg = opt.lambda_reg * (diffRender.calc_reg_loss(Ae) +  diffRender.calc_reg_loss(Ai)) / 2.0
    lossR_flip = opt.lambda_flipz * (diffRender.recon_flip(Ae, L1 = opt.flipL1) + diffRender.recon_flip(Ai, L1 = opt.flipL1) + diffRender.recon_flip(Aire, L1 = opt.flipL1)) / 3.0
    # point not too close
    if opt.lambda_edge>0:
        lossR_reg += opt.lambda_edge * (diffRender.calc_reg_edge(Ae['vertices']) +  diffRender.calc_reg_edge(Ai['vertices'])) / 2.0
    if opt.lambda_depth>0: # z^2
        lossR_reg += opt.lambda_depth * (diffRender.calc_reg_depth(Ae['vertices']) +  diffRender.calc_reg_depth(Ai['vertices'])) / 2.0
    if opt.lambda_depthR>0: #  z^2 * exp (x^2+(y/ratio)^2)
        lossR_reg += opt.lambda_depthR * (diffRender.calc_reg_depthR(Ae['vertices'], temp = opt.temp) +  diffRender.calc_reg_depthR(Ai['vertices'], temp=opt.temp) ) / 2.0
    if opt.lambda_depthC>0: #  z^2 * exp (x^2+(y/ratio)^2)
        lossR_reg += opt.lambda_depthC * (diffRender.calc_reg_depthC(Ae['vertices']) +  diffRender.calc_reg_depthC(Ai['vertices'])) / 2.0
    if opt.lambda_deform>0:
        lossR_reg += opt.lambda_deform* (diffRender.calc_reg_deform(Ae['delta_vertices']) +  diffRender.calc_reg_deform(Ai['delta_vertices'])) / 2.0

    # interpolated cycle consistency. IC need warmup
    #if epoch>=opt.warm_epoch: # Ai is not good at the begining.
    loss_cam, loss_shape, loss_texture, loss_light, loss_bias = diffRender.recon_att(Aire, deep_copy(Ai, detach=True), L1 = opt.L1, chamfer = opt.chamfer, azim = opt.azim)
    lossR_IC = opt.lambda_ic * (loss_cam + loss_shape + loss_texture + loss_light+loss_bias)
    return lossR_reg, lossR_flip, lossR_IC


def trainer(opt, train_dataloader, test_dataloader, train_noaug_dataloader):
    # backup trainer.py and networks.py
    os.system('cp trainer.py %s'%opt.outf)
    os.system('cp networks.py %s'%opt.outf)

    diffRender = DiffRender(mesh_name=opt.template_path, image_size=opt.imageSize, ratio = opt.ratio, init_ellipsoid = opt.ellipsoid, image_weight=opt.image_weight, lambda_lpl = opt.lambda_lpl, lambda_flat = opt.lambda_flat) #for market
    #save_mesh('init.obj', diffRender.vertices_init, template_file.faces, template_file.uvs)

    # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
    netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                            azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                            nc=4, nk=opt.nk, nf=opt.nf, ratio=opt.ratio, makeup=opt.makeup, bg = opt.bg, 
                            pretraint = opt.pretraint, pretrainc = opt.pretrainc, pretrains = opt.pretrains, 
                            droprate = opt.droprate, romp = opt.romp, 
                            coordconv = opt.coordconv, norm = opt.norm, lpl = diffRender.vertices_laplacian_matrix,
                            nolpl = opt.nolpl, inv = opt.inv) # height = 2 * width

    if opt.multigpus:
        netE = torch.nn.DataParallel(netE)
    netE = netE.cuda()

    if opt.fp16:
        scaler = torch.cuda.amp.GradScaler()
    # init template delta
    last_delta_vertices = torch.zeros(diffRender.vertices_init.shape[0], 3).cuda()
    # netL: for Landmark Consistency
    # print(diffRender.num_faces) # 1280
    if opt.lambda_lc>0:
        netL = Landmark_Consistency(num_landmarks=diffRender.num_faces, dim_feat=256, num_samples=64)
        if opt.multigpus:
            netL = torch.nn.DataParallel(netL)
        netL = netL.cuda()

    # netD: Discriminator rgb+seg
    if opt.unmask == 2: # four channel 
        dis_nc = 4 
    else: 
        dis_nc = 3

    if opt.gan_type == 'wgan':
        netD = Discriminator(nc=dis_nc, nf=16)
    elif opt.gan_type == 'lsgan':
        netD = MS_Discriminator(nc=dis_nc, nf=16)
    else:
        print('unknow gan type. Only lsgan or wgan is accepted.')

    if opt.multigpus:
        netD = torch.nn.DataParallel(netD)
    netD = netD.cuda()

    # setup optimizer
    optimizer = optim._multi_tensor.Adam  #https://github.com/huggingface/transformers/issues/9965
    if opt.adamw:
        optimizer = optim._multi_tensor.AdamW

    ignored_params = list(map(id, netE.shape_enc.encoder1.parameters() ))
    add_params = filter(lambda p: id(p) not in ignored_params, netE.parameters())
    pre_params = netE.shape_enc.encoder1.parameters()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd, amsgrad=opt.amsgrad)
    if opt.lambda_lc>0:
        optimizerE = optimizer(list(netE.parameters()) + list(netL.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd, amsgrad=opt.amsgrad)
    else:
        optimizerE = optimizer([             
             {'params': pre_params, 'lr': 0.05 * opt.lr},
             {'params': add_params, 'lr': opt.lr}
              ], betas=(opt.beta1, 0.999), weight_decay=opt.wd, amsgrad=opt.amsgrad)

    # setup learning rate scheduler
    if opt.scheduler == 'step':
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size = round(0.8*opt.niter), gamma=opt.gamma)
        schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE, step_size = round(0.8*opt.niter), gamma=opt.gamma)
    elif  opt.scheduler == 'restart':
        T_0 = opt.niter//(1+2+4)+1
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0 = T_0, T_mult=2, eta_min=opt.gamma*opt.lr)
        schedulerE = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerE, T_0 = T_0, T_mult=2, eta_min=opt.gamma*opt.lr)
    elif  opt.scheduler == 'restart2':
        T_0 = opt.niter//(1+2)+1
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0 = T_0, T_mult=2, eta_min=opt.gamma*opt.lr)
        schedulerE = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerE, T_0 = T_0, T_mult=2, eta_min=opt.gamma*opt.lr)
    elif  opt.scheduler == 'restart1':
        T_0 = int(opt.niter/(1+1))+1
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0 = T_0, T_mult=1, eta_min=opt.gamma*opt.lr)
        schedulerE = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerE, T_0 = T_0, T_mult=1, eta_min=opt.gamma*opt.lr)
    elif  opt.scheduler == 'exp':
        schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.997)
        schedulerE = torch.optim.lr_scheduler.ExponentialLR(optimizerE, gamma=0.997)
    else:
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=opt.niter, eta_min=opt.gamma*opt.lr)
        schedulerE = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE, T_max=opt.niter, eta_min=opt.gamma*opt.lr)

    if opt.swa:
         swa_modelE = AveragedModel(netE).cpu()
         #swa_schedulerE = SWALR(optimizerE, swa_lr=opt.swa_lr)

    # if resume is True, restore from latest_ckpt.path
    start_iter = 0
    start_epoch = 0
    if opt.resume:
        resume_path = os.path.join(opt.outf, 'ckpts/latest_ckpt.pth')
        if os.path.exists(resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            start_iter = 0
            netD.load_state_dict(checkpoint['netD'])
            netE.load_state_dict(checkpoint['netE'])

            optimizerD.load_state_dict(checkpoint['optimizerD'])
            optimizerE.load_state_dict(checkpoint['optimizerE'])

            if opt.swa and start_epoch >= opt.swa_start:
                try:
                    swa_modelE.load_state_dict(checkpoint['swa_modelE'])
                    #swa_schedulerE.load_state_dict(checkpoint['swa_schedulerE'])
                except:
                    print("=> swa model not found")

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))
        else:
            start_iter = 0
            start_epoch = 0
            print("=> no checkpoint can be found")


    ori_dir = os.path.join(opt.outf, 'fid/ori')
    rec_dir = os.path.join(opt.outf, 'fid/rec')
    inter_dir = os.path.join(opt.outf, 'fid/inter')
    inter90_dir = os.path.join(opt.outf, 'fid/inter90')
    ori_mask_dir = os.path.join(opt.outf, 'fid/ori_mask')
    rec_mask_dir = os.path.join(opt.outf, 'fid/rec_mask')
    ckpt_dir = os.path.join(opt.outf, 'ckpts')
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(inter90_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ori_mask_dir, exist_ok=True)
    os.makedirs(rec_mask_dir, exist_ok=True)

    # fid for save
    best_fid = 9999

    summary_writer = SummaryWriter(os.path.join(opt.outf + "/logs"))
    output_txt = './log/%s/result.txt'%opt.name
    warm_up = 0.01 # We start from the 0.01*lrRate
    warm_up_ic = 0.1 # We start from the 0.1*lrRate for ic loss
    warm_iteration = len(train_dataloader)*opt.warm_epoch # first 20 epoch
    print('Model will warm up in %d iterations'%warm_iteration)
    for epoch in range(start_epoch, opt.niter+1):
        for iter, data in enumerate(train_dataloader):
            if epoch<opt.warm_epoch: # 0-19
                warm_up = min(1.0, warm_up + 0.99 / warm_iteration)
            with Timer("Elapsed time in update: %f"):
                ############################
                # (1) Update D network
                ###########################
                optimizerD.zero_grad()
                Xa = data['data']['images'].cuda(non_blocking=True).detach()
                if opt.hmr>0.0:
                    Va = data['data']['obj'].cuda(non_blocking=True).detach()
                img_path = data['data']['path']
                #Ea = data['data']['edge'].cuda()
                batch_size = Xa.shape[0]

                # encode real
                # 0 for train both camera and shape
                if opt.update_shape >0:
                    train_shape = 1 # fix shape train camera
                    if iter % opt.update_shape == 0:
                        train_shape = 2 # fix camera train shape
                else: 
                    train_shape = 0 # train all encoders

                if opt.update_shape ==-1: #a new encoder updating policy
                     if iter % 3 == 0:
                         train_shape = 3 # fix camera + texture,  train shape
                     elif iter % 3 == 1:
                         train_shape = 4 # fix camera + shape, train texture
                     elif iter % 3 == 2:
                         train_shape = 5 # fix shape + texture, train camera

                if opt.fp16:
                    with torch.cuda.amp.autocast():
                        Ae = netE(Xa, need_feats=(opt.lambda_lc>0), img_pth = img_path, train_shape = train_shape )
                else:
                    Ae = netE(Xa, need_feats=(opt.lambda_lc>0), img_pth = img_path, train_shape = train_shape )
                Xer, Ae = diffRender.render(**Ae, no_mask = opt.bg)

                # hard
                if opt.hard:
                    Ae90 = deep_copy(Ae)
                    #Ae90['azimuths'] = - torch.empty((batch_size), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
                    if  random.random()>0.5:
                        Ae90['azimuths'] = - torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(opt.hard_range, 180-opt.hard_range)
                    else:
                        Ae90['azimuths'] = - torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(0, 180)
                    rand = torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(-1.0, 1.0)
                    rand[rand<0] = -1.0
                    rand[rand>=0] = 1.0
                    Ae90['azimuths'] *= rand
                    #print( Ae90['azimuths'])

                # bad case
                mean_delta = torch.mean(torch.abs(Ae['delta_vertices'])[:,-1], dim = 1)
                bad_index = np.argwhere(mean_delta.data.cpu().numpy()>0.4) 
                #print('Collapse index:', bad_index)
                rand_a = np.random.permutation(batch_size)
                rand_b = np.random.permutation(batch_size)
                if opt.inv==0: # inv smooth is used, so here we take care the large movements.
                    good_index = np.setdiff1d( np.arange(batch_size), bad_index) 
                    for i in bad_index: # resample good index 
                        bad_indexa = np.argwhere(rand_a == i)
                        rand_a[bad_indexa] = np.random.choice(good_index, 1)
                        bad_indexb = np.argwhere(rand_b == i)
                        rand_b[bad_indexb] = np.random.choice(good_index, 1)
                rand_a = torch.LongTensor(rand_a)
                rand_b = torch.LongTensor(rand_b)
                Aa = deep_copy(Ae, rand_a)
                Ab = deep_copy(Ae, rand_b)
                Ai = {}
                # linearly interpolate 3D attributes
                if opt.lambda_ic > 0.0:
                    # camera interpolation
                    alpha_camera = torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(0.0, 1.0)
                    Ai['azimuths'] = - torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(-opt.azi_scope/2, opt.azi_scope/2)

                    Ai['elevations'] = torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(netE.camera_enc.elev_min, netE.camera_enc.elev_max)
                    Ai['distances'] = torch.empty((batch_size), dtype=torch.float32, device='cuda').uniform_(netE.camera_enc.dist_min, netE.camera_enc.dist_max)
                    Ai['biases'] = torch.empty((batch_size, 2), dtype=torch.float32, device='cuda').uniform_(-opt.bias_range, opt.bias_range)
                    # texture & shape interpolation
                    if opt.beta>0:
                        beta = min(1.0, opt.beta) # + 0.8*epoch/opt.niter)
                        alpha = torch.FloatTensor(np.random.beta(beta, beta, batch_size), device='cuda')
                        alpha_texture = alpha.view(batch_size, 1, 1, 1)
                        #torch.FloatTensor(np.random.beta(beta, beta, batch_size))
                        alpha = 1-alpha  # to use different  shape and texture pair on purporse
                        alpha_shape = alpha.view(batch_size, 1, 1 )
                    else:
                        alpha_texture = torch.empty((batch_size, 1, 1, 1), dtype=torch.float32, device='cuda').uniform_(0.0, 1.0)
                        alpha_shape = torch.empty((batch_size, 1, 1), dtype=torch.float32, device='cuda').uniform_(0.0, 1.0)

                    Ai['vertices'] = alpha_shape * Aa['vertices'] + (1-alpha_shape) * Ab['vertices']
                    Ai['delta_vertices'] = alpha_shape * Aa['delta_vertices'] + (1-alpha_shape) * Ab['delta_vertices']
                    Ai['textures'] = alpha_texture * Aa['textures'] + (1.0 - alpha_texture) * Ab['textures']
                    if opt.bg:
                        Ai['bg'] = alpha_texture * Aa['bg'] + (1.0 - alpha_texture) * Ab['bg']
                    else:
                        Ai['bg'] = None
                    # light interpolation
                    alpha_light = torch.empty((batch_size, 1), dtype=torch.float32, device='cuda').uniform_(0.0, 1.0)
                    Ai['lights'] = alpha_light * Aa['lights'] + (1.0 - alpha_light) * Ab['lights']
                else:
                    Ai = Ae

                # interpolated 3D attributes render images, and update Ai
                Xir, Ai = diffRender.render(**Ai, no_mask = opt.bg)
                if opt.hard:
                    Xer90, Ae90 = diffRender.render(**Ae90, no_mask = opt.bg)
                else:
                    Xer90 = Xer
                # save img to a temporal dir
                #tmp_path = [] 
                #for i in range(len(img_path)):
                #    path = img_path[i]
                #    image_name = os.path.basename(path)
                #    inter_path = os.path.join(inter_dir, image_name)
                #    output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
                #    output_Xir.save(inter_path, 'JPEG', quality=100)

                #if opt.fp16:
                #    with torch.cuda.amp.autocast():
                        # predicted 3D attributes from above render images
                #        Aire = netE(Xir.detach().clone(), need_feats=(opt.lambda_lc>0), train_shape = 0 ) #, img_pth = tmp_path)
                #else:
                    # predicted 3D attributes from above render images 
                Aire = netE(Xir.detach().clone(), need_feats=(opt.lambda_lc>0), train_shape = 0 ) #, img_pth = tmp_path)
                # render again to update predicted 3D Aire 
                _, Aire = diffRender.render(**Aire, no_mask = opt.bg)

                # discriminate loss
                if opt.unmask == 1: 
                    Ma = Xa[:,:3]
                    Mer90 = Xer90[:,:3]
                    Mir = Xir[:,:3]
                elif opt.unmask == 0:
                    Ma = mask(Xa)
                    Mer90 = mask(Xer90)
                    Mir = mask(Xir)
                elif opt.unmask == 2:
                    Ma, Mer90, Mir = Xa, Xer90, Xir
                else:
                    print('Please specify unmask')
                #outs0 = netD(Ma.detach().clone()) # real
                #outs1 = netD(Mer90.detach().clone()) # fake - recon?
                #outs2 = netD(Mir.detach().clone()) # fake - inter?
                ##########################
                # Since no batch norm is used in Discriminator, there will be no side effect to 
                # forward Ma, Ner90, Mir together
                ##########################
                if opt.fp16:
                    with torch.cuda.amp.autocast():
                        outs = netD( torch.cat( (Ma.detach().clone(), Mer90.detach().clone(), Mir.detach().clone()), dim=0))
                else:
                    outs = netD( torch.cat( (Ma.detach().clone(), Mer90.detach().clone(), Mir.detach().clone()), dim=0))
                lossD, lossD_real, lossD_fake, lossD_gp  = 0, 0, 0, 0
 
                if opt.gan_type == 'wgan': # WGAN-GP
                    outs0, outs1,outs2 = torch.split(outs, batch_size, dim= 0)
                    lossD_real = opt.lambda_gan * torch.mean(outs0)
                    lossD_fake = opt.lambda_gan * ( torch.mean(outs1) + opt.ganw*torch.mean(outs2)) / (1.0+opt.ganw)

                    lossD_gp = opt.gan_reg * opt.lambda_gan * (compute_gradient_penalty(netD, Ma.data, Mer90.data) + \
                                        opt.ganw*compute_gradient_penalty(netD, Ma.data, Mir.data)) / (1.0+opt.ganw)
                    lossD = lossD_fake - lossD_real + lossD_gp
                elif opt.gan_type == 'lsgan':
                    for it, out in enumerate(outs):
                        out0, out1,out2 = torch.split(out, batch_size, dim= 0)
                        lossD_real += opt.lambda_gan * torch.mean((out0 - 1)**2)
                        lossD_fake += opt.lambda_gan * (torch.mean((out1 - 0)**2) + opt.ganw*torch.mean((out2 - 0)**2)) /(1.0+opt.ganw)
                    lossD_gp = opt.gan_reg * opt.lambda_gan * (compute_gradient_penalty_list(netD, Ma.data, Mer90.data) + \
                                    opt.ganw*compute_gradient_penalty_list(netD, Ma.data, Mir.data)) / (1.0+opt.ganw)
                    lossD = lossD_fake + lossD_real + lossD_gp 
                lossD *= warm_up
                if opt.fp16:
                    scaler.scale(lossD).backward()
                    scaler.step(optimizerD)
                else:
                    lossD.backward()
                    optimizerD.step()

                ############################
                # (2) Update G network
                # fix netE.shape to update 
                ###########################
                optimizerE.zero_grad()
                # GAN loss
                lossR_fake = 0
                if opt.fp16:
                    with torch.cuda.amp.autocast():
                        outs = netD(torch.cat( (Mer90, Mir), dim=0))
                else:
                    outs = netD(torch.cat( (Mer90, Mir), dim=0))
                if opt.gan_type == 'wgan':
                    outs1,outs2 = torch.split(outs, batch_size, dim= 0)
                    lossR_fake = opt.lambda_gan * (-outs1.mean() - opt.ganw*outs2.mean()) / (1.0+opt.ganw)
                elif opt.gan_type == 'lsgan':
                    for it, out in enumerate(outs):
                        out1,out2 = torch.split(out, batch_size, dim= 0)
                        lossR_fake += opt.lambda_gan * ( torch.mean((out1 - 1)**2) + opt.ganw*torch.mean((out2 - 1)**2)) / (1.0+opt.ganw)

                # Image Recon loss.
                lossR_data = opt.lambda_data * diffRender.recon_data(Xer, Xa, no_mask = opt.bg, contour = opt.lambda_contour)

                if opt.hmr > 0:
                    #print(Ae['vertices'].shape, Va.shape)
                    cham_dist, cham_normals = chamfer_distance(Ae['vertices'], Va)
                    lossR_data += opt.hmr * cham_dist

                # mesh regularization
                if opt.fp16:
                    with torch.cuda.amp.autocast():
                        lossR_reg, lossR_flip, lossR_IC = regularization(diffRender, Ae, Ai, Aire, opt)
                else:
                    lossR_reg, lossR_flip, lossR_IC = regularization(diffRender, Ae, Ai, Aire, opt)
                # disentangle regularization
                lossR_dis = 0.0
                if opt.dis1>0 or opt.dis2>0:
                    bnum = Ae['vertices'].shape[0]
                    # change camera & light direction, keep shape and texture
                    if opt.dis1>0:
                        if opt.fp16:
                            with torch.cuda.amp.autocast():
                                Ae_fliplr = netE(fliplr(Xa), need_feats=False, img_pth = img_path)
                        else:
                            Ae_fliplr = netE(fliplr(Xa), need_feats=False, img_pth = img_path)
                        l_text = torch.abs( fliplr(Ae_fliplr['textures']) - Ae['textures']).mean()
                        Na = Ae['vertices'].clone()
                        Na[..., 0] *=-1 # flip x 
                        if opt.chamfer:
                            l_shape, _  = chamfer_distance(Ae_fliplr['vertices'], Na)
                        else: #L2 loss
                            l_shape = torch.norm(Ae_fliplr['vertices'].view(bnum,-1) - Na.view(bnum,-1), p=2, dim=1).mean()
                        lossR_dis += opt.dis1 * (l_text + l_shape)
                    # change texture, keep camera and shape
                    # jitter = ColorJitter(brightness=.5, hue=.3)
                    if opt.dis2>0:
                        re = torchvision.transforms.RandomErasing(p=1)
                        if opt.fp16:
                            with torch.cuda.amp.autocast():
                                Ae_jitter = netE(re(Xa), need_feats=False, img_pth = img_path)
                        else:
                            Ae_jitter = netE(re(Xa), need_feats=False, img_pth = img_path)
                        if opt.chamfer:
                            l_shape, _  = chamfer_distance(Ae_jitter['vertices'],  Ae['vertices'])
                        else: #L2 loss
                            l_shape = torch.norm(Ae_jitter['delta_vertices'].view(bnum,-1) - Ae['delta_vertices'].view(bnum,-1), p=2, dim=1).mean()
                        loss_azim = torch.pow(angle2xy(Ae_jitter['azimuths']) -
                             angle2xy(Ae['azimuths']), 2).mean()
                        loss_elev = torch.pow(angle2xy(Ae_jitter['elevations']) -
                             angle2xy(Ae['elevations']), 2).mean()
                        loss_dist = torch.pow(Ae_jitter['distances'] - Ae['distances'], 2).mean()
                        loss_bias = torch.pow(Ae_jitter['biases'] - Ae['biases'], 2).mean()
                        l_cam = opt.azim * loss_azim + loss_elev + loss_dist + loss_bias
                    #l_light = 0.1  * torch.pow(Ae_jitter['lights'] - Ae['lights'], 2).mean()
                        lossR_dis += opt.dis2 * (l_cam + l_shape)

                # landmark consistency
                if opt.lambda_lc>0:
                    Le = Ae['faces_image']
                    Li = Aire['faces_image']
                    Fe = Ae['img_feats']
                    Fi = Aire['img_feats']
                    Ve = Ae['visiable_faces']
                    Vi = Aire['visiable_faces']
                    lossR_LC = opt.lambda_lc * (netL(Fe, Le, Ve).mean() + netL(Fi, Li, Vi).mean())
                else:
                    lossR_LC = 0.0

                # overall loss
                lossR = lossR_fake + lossR_reg + lossR_flip  + lossR_data + lossR_IC +  lossR_LC + lossR_dis

                lossR *= warm_up
                if opt.fp16:
                    scaler.scale(lossR).backward()
                    scaler.step(optimizerE)
                    scaler.update()
                else:
                    lossR.backward()
                    optimizerE.step()

                print('Name: ', opt.outf)
                print('[%d/%d][%d/%d]\n'
                'LossD: %.4f lossD_real: %.4f lossD_fake: %.4f lossD_gp: %.4f\n'
                'lossR: %.4f lossR_fake: %.4f lossR_reg: %.4f lossR_data: %.4f '
                'lossR_IC: %.4f lossR_dis: %.4f \n'
                    % (epoch, opt.niter, iter, len(train_dataloader),
                        lossD, lossD_real, lossD_fake, lossD_gp,
                        lossR, lossR_fake, lossR_reg, lossR_data,
                        lossR_IC, lossR_dis
                        )
                ) 
                del lossD, lossD_real, lossD_fake, lossD_gp, lossR, lossR_fake, lossR_reg, lossR_data, lossR_IC, lossR_dis 
        if opt.swa and epoch >= opt.swa_start and epoch%opt.swa_interval==0:
            swa_modelE.cuda()
            swa_modelE.update_parameters(netE)
            swa_modelE.cpu()
            print('How many models arer fused: %d'%swa_modelE.n_averaged)
            #swa_schedulerE.step()
        schedulerD.step()
        schedulerE.step()

        ############################
        # Evaluate Model
        ###########################
        Xa_clone =  Xa.clone()

        if epoch % 10 == 0:  
            print('===========Saving JPEG===========')
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

            randperm_a = torch.randperm(batch_size)
            randperm_b = torch.randperm(batch_size)

            vutils.save_image(Xa[randperm_a, :3],
                    '%s/epoch_%03d_Iter_%04d_randperm_Xa.png' % (opt.outf, epoch, iter), normalize=True)
            vutils.save_image(Xa[randperm_a, :3],
                    '%s/current_randperm_Xa.png' % (opt.outf), normalize=True)

            vutils.save_image(Xa[randperm_b, :3],
                    '%s/epoch_%03d_Iter_%04d_randperm_Xb.png' % (opt.outf, epoch, iter), normalize=True)
            vutils.save_image(Xa[randperm_b, :3],
                    '%s/current_randperm_Xb.png' % (opt.outf), normalize=True)

            vutils.save_image(Xa[:, :3],
                    '%s/epoch_%03d_Iter_%04d_Xa.png' % (opt.outf, epoch, iter), normalize=True)
            vutils.save_image(Xa[:, :3],
                    '%s/current_Xa.png' % (opt.outf), normalize=True)

            vutils.save_image(Xer[:, :3].detach(),
                    '%s/epoch_%03d_Iter_%04d_Xer.png' % (opt.outf, epoch, iter), normalize=True)
            vutils.save_image(Xer[:, :3].detach(),
                    '%s/current_Xer.png' % (opt.outf), normalize=True)

            vutils.save_image(Xir[:, :3].detach(),
                    '%s/epoch_%03d_Iter_%04d_Xir.png' % (opt.outf, epoch, iter), normalize=True)
            vutils.save_image(Xir[:, :3].detach(),
                    '%s/current_Xir.png' % (opt.outf), normalize=True)

            vutils.save_image(textures.detach(),
                    '%s/current_textures.png' % (opt.outf), normalize=True)

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

            #tri_mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy())
            #tri_mesh.export('%s/current_mesh_recon.obj' % opt.outf)
            #tri_mesh.export('%s/epoch_%03d_mesh_recon.obj' % (opt.outf, epoch))
            save_mesh('%s/current_mesh_recon.obj' % opt.outf, vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)
            #save_mesh('%s/epoch_%03d_mesh_recon.obj' % (opt.outf, epoch), vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)
            save_mesh('%s/epoch_%03d_template.obj' % (opt.outf, epoch), netE.vertices_init[0].clone().detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)

            print('===========Saving Gif-Azi===========')
            rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation.gif' % epoch)
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(range(-int(opt.azi_scope/2), int(opt.azi_scope/2), 10))) # -180, 180
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')
            for delta_azimuth in loop:
                Ae['azimuths'] = - torch.tensor([delta_azimuth], dtype=torch.float32, device='cuda').repeat(batch_size)
                predictions, _ = diffRender.render(**Ae)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()
            current_rotate_path = os.path.join(opt.outf, 'current_rotation.gif')
            shutil.copyfile(rotate_path, current_rotate_path)

            print('===========Saving Gif-Y===========')
            rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation_ele.gif' % epoch)
            writer = imageio.get_writer(rotate_path, mode='I')
            elev_range = opt.elev_range.split('~')
            elev_min = int(elev_range[0])
            elev_max = int(elev_range[1])
            loop = tqdm.tqdm(list(range(elev_min, elev_max, 10))) # 15~-45
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_ele)')
            for delta_elevation in loop:
                Ae['elevations'] = - torch.tensor([delta_elevation], dtype=torch.float32, device='cuda').repeat(batch_size)
                predictions, _ = diffRender.render(**Ae)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()
            current_rotate_path = os.path.join(opt.outf, 'current_rotation_ele.gif')
            shutil.copyfile(rotate_path, current_rotate_path)

            print('===========Saving Gif-Dist===========')
            rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation_dist.gif' % epoch)
            writer = imageio.get_writer(rotate_path, mode='I')
            dist_range = opt.dist_range.split('~')
            dist_min = int(dist_range[0])
            dist_max = int(dist_range[1])
            loop = tqdm.tqdm(list(range(dist_min, dist_max+1, 1))) # 1, 7
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_dist)')
            for delta_dist in loop:
                Ae['distances'] = - torch.tensor([delta_dist], dtype=torch.float32, device='cuda').repeat(batch_size)
                predictions, _ = diffRender.render(**Ae)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()
            current_rotate_path = os.path.join(opt.outf, 'current_rotation_dist.gif')
            shutil.copyfile(rotate_path, current_rotate_path)

        if opt.swa and epoch >= opt.swa_start and epoch % 20 == 0:
            print('===========Updating SWA BatchNorm===========')
            swa_modelE.cuda()
            update_bn(train_dataloader, swa_modelE)
            swa_modelE.eval()

            swa_Ae = swa_modelE(Xa_clone, need_feats=(opt.lambda_lc>0), img_pth = img_path, train_shape = train_shape )
            print('===========Saving Gif-Azi===========')
            rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation_swa.gif' % epoch)
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(range(-int(opt.azi_scope/2), int(opt.azi_scope/2), 10))) # -180, 180
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics (Gif_azi)')
            for delta_azimuth in loop:
                swa_Ae['azimuths'] = - torch.tensor([delta_azimuth], dtype=torch.float32, device='cuda').repeat(batch_size)
                predictions, _ = diffRender.render(**swa_Ae)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()
            current_rotate_path = os.path.join(opt.outf, 'current_rotation_swa.gif')
            shutil.copyfile(rotate_path, current_rotate_path)

        if epoch % 20 == 0: # and epoch > 0:
            print('===========Generating Test Images===========')
            netE.eval()
            X_all = []
            path_all = []
            for i, data in tqdm.tqdm(enumerate(test_dataloader)):
                Xa = data['data']['images'].cuda()
                paths = data['data']['path']

                with torch.no_grad():
                    Ae = netE(Xa)
                    Xer, Ae = diffRender.render(**Ae)

                    Ai = deep_copy(Ae)
                    Ai2 = deep_copy(Ae)
                    Ae90 = deep_copy(Ae)
                    Ae270 = deep_copy(Ae)
                    Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32, device='cuda').uniform_(-opt.azi_scope/2, opt.azi_scope/2)
                    Ai2['azimuths'] = Ai['azimuths'] + 90.0 # random+90
                    Ai2['azimuths'][Ai2['azimuths']>180] -= 360.0 # -180, 180
                    Ae90['azimuths'] += 90.0 # recon+90
                    Ae270['azimuths'] -= 90.0 # recon-90

                    Xir, Ai = diffRender.render(**Ai)
                    Xir2, _ = diffRender.render(**Ai2)
                    Xer90, _ = diffRender.render(**Ae90)
                    Xer270, _ = diffRender.render(**Ae270)


                    # multiprocess to save image
                    for i in range(len(paths)):
                        path = paths[i]
                        image_name = os.path.basename(path)
                        rec_path = os.path.join(rec_dir, image_name)
                        output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
                        #output_Xer.save(rec_path, 'JPEG', quality=100)

                        inter_path = os.path.join(inter_dir, image_name)
                        output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
                        #output_Xir.save(inter_path, 'JPEG', quality=100)

                        inter_path2 = os.path.join(inter_dir, '2+'+image_name)
                        output_Xir2 = to_pil_image(Xir2[i, :3].detach().cpu())
                        #output_Xir2.save(inter_path2, 'JPEG', quality=100)

                        inter90_path = os.path.join(inter90_dir, image_name)
                        output_Xer90 = to_pil_image(Xer90[i, :3].detach().cpu())
                        #output_Xer90.save(inter90_path, 'JPEG', quality=100)

                        inter270_path = os.path.join(inter90_dir, '2+'+image_name)
                        output_Xer270 = to_pil_image(Xer270[i, :3].detach().cpu())

                        rec_mask_path = os.path.join(rec_mask_dir, image_name)
                        output_rec_mask = to_pil_image(Xer[i, 3].detach().cpu())

                        if epoch==0:
                            ori_path = os.path.join(ori_dir, image_name)
                            if opt.bg:
                                gt_img = Xa[:, :3]
                                gt_mask = Xa[:, 3]
                                Xa[:, :3] = gt_img * gt_mask.unsqueeze(1) + torch.ones_like(gt_img) * (1 - gt_mask.unsqueeze(1))
                            output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
                            ori_mask_path = os.path.join(ori_mask_dir, image_name)
                            output_ori_mask =  to_pil_image(Xa[i, 3].detach().cpu())
                            #output_Xa.save(ori_path, 'JPEG', quality=100)
                            X_all.extend([output_Xa, output_ori_mask])
                            path_all.extend([ori_path, ori_mask_path])

                        X_all.extend([output_Xer, output_Xir, output_Xir2, output_Xer90, output_Xer270, output_rec_mask])
                        path_all.extend([rec_path, inter_path, inter_path2, inter90_path, inter270_path, rec_mask_path])
            # save image
            with Pool(4) as p:
                p.map(save_img, zip(X_all, path_all) )        

            print('===========Evaluating SSIM & MaskIoU===========')
            ssim_score = []
            mask_score = []
            for root, dirs, files in os.walk(ori_dir, topdown=True):
                for name in files:
                    if not ( name[-3:]=='png' or name[-3:]=='jpg'):
                        continue
                    # SSIM
                    ori_path = ori_dir + '/' + name
                    rec_path = rec_dir + '/' + name
                    ori = Image.open(ori_path).convert('RGB').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    rec = Image.open(rec_path).convert('RGB').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    ori = torchvision.transforms.functional.to_tensor(ori).unsqueeze(0)
                    rec = torchvision.transforms.functional.to_tensor(rec).unsqueeze(0)
                    ssim_score.append(ssim(ori, rec, data_range=1))
                    # Mask IoU
                    ori_path = ori_mask_dir + '/' + name
                    rec_path = rec_mask_dir + '/' + name
                    ori = Image.open(ori_path).convert('L').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    rec = Image.open(rec_path).convert('L').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    ori = torchvision.transforms.functional.to_tensor(ori)
                    rec = torchvision.transforms.functional.to_tensor(rec)
                    mask_score.append(1 - mask_iou(ori, rec)) # the default mask iou is maskiou loss. https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/render.py. So we have to 1- maskiou loss to obtain the mask iou

            print('\033[1mTest recon ssim: %0.3f \033[0m' % np.mean(ssim_score) )
            print('\033[1mTest recon MaskIoU: %0.3f\033[0m' % np.mean(mask_score) )

            print('===========Evaluating FID Score===========')
            fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 64, True)
            print('Epoch %03d Test recon fid: %0.2f' % (epoch, fid_recon) ) 
            summary_writer.add_scalar('Test/fid_recon', fid_recon, epoch)
            fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 64, True)
            print('Epoch %03d Test rotation fid: %0.2f' % (epoch, fid_inter))
            summary_writer.add_scalar('Test/fid_inter', fid_inter, epoch)
            fid_90 = calculate_fid_given_paths([ori_dir, inter90_dir], 64, True)
            print('Epoch %03d Test rotat90/270 fid: %0.2f' % (epoch, fid_90))
            summary_writer.add_scalar('Test/fid_90', fid_90, epoch)
            with open(output_txt, 'a') as fp:
                fp.write('Epoch %03d recon ssim: %0.3f\n' % (epoch, np.mean(ssim_score)))
                fp.write('Epoch %03d recon MaskIoU: %0.3f\n' % (epoch, np.mean(mask_score)))
                fp.write('Epoch %03d Test recon fid: %0.2f\n' % (epoch, fid_recon))
                fp.write('Epoch %03d Test rotation fid: %0.2f\n' % (epoch, fid_inter))
                fp.write('Epoch %03d Test rotate90/270 fid: %0.2f\n' % (epoch, fid_90))

            print('===========Saving Best Snapshot===========')
            epoch_name = os.path.join(ckpt_dir, 'epoch_%05d.pth' % epoch)
            latest_name = os.path.join(ckpt_dir, 'latest_ckpt.pth')
            best_name = os.path.join(ckpt_dir, 'best_ckpt.pth')
            best_mesh_name = os.path.join(ckpt_dir, 'best_mesh.obj')
            state_dict = {
                'epoch': epoch,
                'netE': netE.state_dict(),
                'netD': netD.state_dict(),
                #'optimizerE': optimizerE.state_dict(),
                #'optimizerD': optimizerD.state_dict()
            }
            if opt.swa and epoch >= opt.swa_start:
                state_dict.update({
                    'swa_modelE': swa_modelE.state_dict(),
                    #'swa_schedulerE': swa_schedulerE.state_dict(),
                })
            torch.save(state_dict, latest_name)
            if fid_inter < best_fid:
                torch.save(state_dict, best_name)
                save_mesh(best_mesh_name, netE.vertices_init[0].clone().detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)
                best_fid = fid_inter 

        if opt.swa and epoch >= opt.swa_start and epoch % 20 == 0:
            print('===========Generating SWA Test Images===========')
            X_all = []
            path_all = []
            for i, data in tqdm.tqdm(enumerate(test_dataloader)):
                Xa = data['data']['images'].cuda()
                paths = data['data']['path']

                with torch.no_grad():
                    Ae = swa_modelE(Xa)
                    Xer, Ae = diffRender.render(**Ae)

                    Ai = deep_copy(Ae)
                    Ai2 = deep_copy(Ae)
                    Ae90 = deep_copy(Ae)
                    Ae270 = deep_copy(Ae)
                    Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32, device='cuda').uniform_(-opt.azi_scope/2, opt.azi_scope/2)
                    Ai2['azimuths'] = Ai['azimuths'] + 90.0
                    Ai2['azimuths'][Ai2['azimuths']>180] -= 360.0 # -180, 180
                    Ae90['azimuths'] += 90.0
                    Ae270['azimuths'] -= 90.0

                    Xir, _ = diffRender.render(**Ai)
                    Xir2, _ = diffRender.render(**Ai2)
                    Xer90, _ = diffRender.render(**Ae90)
                    Xer270, _ = diffRender.render(**Ae270)


                    # multiprocess to save image
                    for i in range(len(paths)):
                        path = paths[i]
                        image_name = os.path.basename(path)
                        rec_path = os.path.join(rec_dir, image_name)
                        output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
                        #output_Xer.save(rec_path, 'JPEG', quality=100)

                        inter_path = os.path.join(inter_dir, image_name)
                        output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
                        #output_Xir.save(inter_path, 'JPEG', quality=100)

                        inter_path2 = os.path.join(inter_dir, '2+'+image_name)
                        output_Xir2 = to_pil_image(Xir2[i, :3].detach().cpu())
                        #output_Xir2.save(inter_path2, 'JPEG', quality=100)

                        inter90_path = os.path.join(inter90_dir, image_name)
                        output_Xer90 = to_pil_image(Xer90[i, :3].detach().cpu())
                        #output_Xer90.save(inter90_path, 'JPEG', quality=100)

                        inter270_path = os.path.join(inter90_dir, '2+'+image_name)
                        output_Xer270 = to_pil_image(Xer270[i, :3].detach().cpu())

                        rec_mask_path = os.path.join(rec_mask_dir, image_name)
                        output_rec_mask = to_pil_image(Xer[i, 3].detach().cpu())

                        if epoch==0:
                            ori_path = os.path.join(ori_dir, image_name)
                            if opt.bg:
                                gt_img = Xa[:, :3]
                                gt_mask = Xa[:, 3]
                                Xa[:, :3] = gt_img * gt_mask.unsqueeze(1) + torch.ones_like(gt_img) * (1 - gt_mask.unsqueeze(1))
                            output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
                            ori_mask_path = os.path.join(ori_mask_dir, image_name)
                            output_ori_mask =  to_pil_image(Xa[i, 3].detach().cpu())
                            #output_Xa.save(ori_path, 'JPEG', quality=100)
                            X_all.extend([output_Xa, output_ori_mask])
                            path_all.extend([ori_path, ori_mask_path])

                        X_all.extend([output_Xer, output_Xir, output_Xir2, output_Xer90, output_Xer270, output_rec_mask])
                        path_all.extend([rec_path, inter_path, inter_path2, inter90_path, inter270_path, rec_mask_path])
            # save image
            with Pool(4) as p:
                p.map(save_img, zip(X_all, path_all) )        

            print('===========Evaluating SWA SSIM & MaskIoU===========')
            ssim_score = []
            mask_score = []
            for root, dirs, files in os.walk(ori_dir, topdown=True):
                for name in files:
                    if not ( name[-3:]=='png' or name[-3:]=='jpg'):
                        continue
                    # SSIM
                    ori_path = ori_dir + '/' + name
                    rec_path = rec_dir + '/' + name
                    ori = Image.open(ori_path).convert('RGB').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    rec = Image.open(rec_path).convert('RGB').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    ori = torchvision.transforms.functional.to_tensor(ori).unsqueeze(0)
                    rec = torchvision.transforms.functional.to_tensor(rec).unsqueeze(0)
                    ssim_score.append(ssim(ori, rec, data_range=1))
                    # Mask IoU
                    ori_path = ori_mask_dir + '/' + name
                    rec_path = rec_mask_dir + '/' + name
                    ori = Image.open(ori_path).convert('L').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    rec = Image.open(rec_path).convert('L').resize((opt.imageSize, round(opt.imageSize*opt.ratio)))
                    ori = torchvision.transforms.functional.to_tensor(ori)
                    rec = torchvision.transforms.functional.to_tensor(rec)
                    mask_score.append(1 - mask_iou(ori, rec)) # the default mask iou is maskiou loss. https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/render.py. So we have to 1- maskiou loss to obtain the mask iou

            print('\033[1m SWA Test recon ssim: %0.3f \033[0m' % np.mean(ssim_score) )
            print('\033[1m SWA Test recon MaskIoU: %0.3f\033[0m' % np.mean(mask_score) )

            print('===========Evaluating FID Score===========')
            fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 64, True)
            print('Epoch %03d SWA Test recon fid: %0.2f' % (epoch, fid_recon) ) 
            summary_writer.add_scalar('Test/fid_recon', fid_recon, epoch)
            fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 64, True)
            print('Epoch %03d SWA Test rotation fid: %0.2f' % (epoch, fid_inter))
            summary_writer.add_scalar('Test/fid_inter', fid_inter, epoch)
            fid_90 = calculate_fid_given_paths([ori_dir, inter90_dir], 64, True)
            print('Epoch %03d SWA Test rotat90 fid: %0.2f' % (epoch, fid_90))
            summary_writer.add_scalar('Test/fid_90', fid_90, epoch)
            with open(output_txt, 'a') as fp:
                fp.write('Epoch %03d recon ssim: %0.3f (SWA) \n' % (epoch, np.mean(ssim_score)))
                fp.write('Epoch %03d recon MaskIoU: %0.3f (SWA) \n' % (epoch, np.mean(mask_score)))
                fp.write('Epoch %03d Test recon fid: %0.2f (SWA) \n' % (epoch, fid_recon))
                fp.write('Epoch %03d Test rotation fid: %0.2f (SWA) \n' % (epoch, fid_inter))
                fp.write('Epoch %03d Test rotate90/270 fid: %0.2f (SWA) \n' % (epoch, fid_90))

            print('===========Saving Best Snapshot===========')
            epoch_name = os.path.join(ckpt_dir, 'epoch_%05d.pth' % epoch)
            latest_name = os.path.join(ckpt_dir, 'latest_ckpt.pth')
            best_name = os.path.join(ckpt_dir, 'best_ckpt.pth')
            best_mesh_name = os.path.join(ckpt_dir, 'best_mesh.obj')
            state_dict = {
                'epoch': epoch,
                'netE': netE.state_dict(),
                'netD': netD.state_dict(),
                #'optimizerE': optimizerE.state_dict(),
                #'optimizerD': optimizerD.state_dict()
            }
            if opt.swa and epoch >= opt.swa_start:
                state_dict.update({
                    'swa_modelE': swa_modelE.state_dict(),
                    #'swa_schedulerE': swa_schedulerE.state_dict(),
                })
            torch.save(state_dict, latest_name)
            if fid_inter < best_fid:
                torch.save(state_dict, best_name)
                save_mesh(best_mesh_name, netE.vertices_init[0].clone().detach().cpu().numpy(), faces.detach().cpu().numpy(), uvs)
                best_fid = fid_inter 

        swa_modelE.cpu()        
        ############################
        # (3) Update Template
        ###########################
        # azimuth 0, mean distance and mIoU of 0.8  with template:
        # template projection
        netE.eval()
        elev_range = opt.elev_range.split('~')
        elev_min = float(elev_range[0])
        elev_max = float(elev_range[1])
        dist_range = opt.dist_range.split('~')
        dist_min = float(dist_range[0])
        dist_max = float(dist_range[1])
        mean_elev = (elev_max + elev_min) /2
        mean_dist = (dist_max + dist_min) /2
        # fix the template for the final swa model
        if opt.em > 0 and epoch%opt.em_gap == 0 and epoch < opt.swa_start: # and epoch<int(0.8*opt.niter):
            print('===========Updating template===========')
            sample_number = len(train_dataloader.dataset)//opt.batchSize * opt.batchSize
            current_delta_vertices = torch.zeros(diffRender.vertices_init.shape[0], 3).cuda()
            all_vertices = torch.zeros(sample_number, diffRender.vertices_init.shape[0], 3) # all
            all_delta_vertices = torch.zeros(sample_number, diffRender.vertices_init.shape[0], 3) # all
            for iter, data in enumerate(train_noaug_dataloader):
                Xa = data['data']['images'].cuda()
                with torch.no_grad():
                    Ae = netE(Xa)
                    _, Ae0 = diffRender.render(**Ae)
                    #Ae = netE(fliplr(Xa))
                    #_, Ae1 = diffRender.render(**Ae)
                    if opt.white:
                        Ae0 = white(Ae0)
                    #    Ae1 = white(Ae1)
                    #Ae0['vertices'] = (Ae0['vertices'] + Ae1['vertices']) / 2.0
                    #Ae0['delta_vertices'] = (Ae0['delta_vertices'] + Ae1['delta_vertices']) / 2.0

                start  = iter * opt.batchSize
                end = start + opt.batchSize
                all_vertices[ start: end , :, :] = Ae0['vertices'].data.cpu()
                all_delta_vertices[ start: end , :, :] = Ae0['delta_vertices'].data.cpu()

            # delete base case with extreme large change
            mean_delta = torch.mean(torch.abs(all_delta_vertices)[:, -1], dim = 1)
            good_index = np.argwhere(mean_delta.data.cpu().numpy()<=0.4)
            all_vertices = all_vertices[good_index, :, :]
            all_delta_vertices = all_delta_vertices[good_index, :, :]
            print('Extreme Bad Case: %d'% (sample_number - len(good_index)) )

            if opt.em == 2: # only poistive
                good_index = torch.mean(all_vertices[:,:,2], dim=1) >= 0.001 # hand is in front of the human by depth.
                current_delta_vertices =  torch.sum(all_delta_vertices[good_index],dim=0)
                count = len(good_index)
            elif opt.em == 3: # symmetry
                left = torch.sum(all_vertices[:,:,0]>0, dim=1)
                front =  torch.sum(all_vertices[:,:,2]>0, dim=1) 
                good_index1 = torch.abs( left - netE.num_vertices //2) < int(netE.num_vertices*0.1)
                good_index2 = torch.abs( front - netE.num_vertices //2) < int(netE.num_vertices*0.1)
                good_index = torch.logical_and(good_index1, good_index2) 
                current_delta_vertices =  torch.sum(all_delta_vertices[good_index],dim=0)
                count = len(good_index)
            elif opt.em == 4: # DBSCAN
                #all_vertices = torch.rand(sample_number, diffRender.vertices_init.shape[0], 3)
                all_vertices = all_vertices.view(sample_number, -1)  
                # white
                all_vertices -= torch.mean(all_vertices, dim=1, keepdim = True)
                all_vertices /= torch.std(all_vertices, dim=1, unbiased=True, keepdim=True) 
                # L2 Norm
                fnorm = torch.norm(all_vertices, p=2, dim=1, keepdim=True) + 1e-8
                all_vertices = all_vertices.div(fnorm.expand_as(all_vertices))
                # cluster
                similarity_metric = torch.mm(all_vertices, all_vertices.transpose(0,1)) 
                similarity_metric[similarity_metric>1] = 1  #due to the epsilon
                dist_metric = 2 - 2*similarity_metric
                clustering = DBSCAN(eps=opt.eps, min_samples= int(sample_number*0.1), metric='precomputed', algorithm='auto').fit(dist_metric.numpy())
                # most frequent 
                valid_cluster = clustering.labels_[ np.argwhere(clustering.labels_ != -1) ]
                if len(valid_cluster)>0:
                    val,counts = np.unique(valid_cluster, return_counts=True)
                    most_fre_index = np.argmax(counts) 
                    good_index = torch.LongTensor( np.argwhere( clustering.labels_ == val[most_fre_index]) )
                    print('Cluster %d is selected!'% val[most_fre_index] )
                    current_delta_vertices =  torch.sum(all_delta_vertices[good_index],dim=0)
                    count = len(good_index)
                else: 
                    print('No good clusters are found! Use all data to update.')
                    current_delta_vertices =  torch.sum(all_delta_vertices,dim=0)
                    count = all_vertices.shape[0]
            elif opt.em ==5: 
                #Eucledian dist
                dist = torch.sum(all_delta_vertices.view(sample_number, -1)**2, dim=1)
                # good_index
                index = np.argsort(dist.cpu().numpy())  #from small to large
                good_index = index[0: int(sample_number*opt.topK)] 
                current_delta_vertices =  torch.sum(all_delta_vertices[good_index],dim=0)
                count = len(good_index) 
            else: # all average
                current_delta_vertices =  torch.sum(all_delta_vertices,dim=0)
                count = all_vertices.shape[0] 
           
            print('The template mesh fuses %d / %d meshes since last batch is droppped'%(count, sample_number) )
            if count > 1:
                #last_delta_vertices = 0.9*last_delta_vertices + 0.1*current_delta_vertices * 1.0 / count 
                last_delta_vertices = current_delta_vertices.cuda() * 1.0 / count 
                if opt.smooth>0:
                    delta_vertices_laplacian = torch.matmul(diffRender.vertices_laplacian_matrix.cuda(), last_delta_vertices)
                    last_delta_vertices += delta_vertices_laplacian* opt.smooth  # move to the middle point towards the neighbor
                    if opt.em >= 6:
                        # more time to know neighbor's neighbor
                        for smooth_times in range(int(opt.em-5)):
                            delta_vertices_laplacian = torch.matmul(diffRender.vertices_laplacian_matrix.cuda(), last_delta_vertices)
                            last_delta_vertices += delta_vertices_laplacian* opt.smooth
                last_delta_vertices[last_delta_vertices>opt.clip] = opt.clip # clip 0.05 == 1/20
                last_delta_vertices[last_delta_vertices<-opt.clip] = - opt.clip # clip
                new_template = netE.vertices_init.data + warm_up*opt.em_step*last_delta_vertices
                #new_template[new_template>0.999] = 0.999
                #new_template[new_template<-0.999] = -0.999
               # 1*642*3
                if opt.white:
                    new_template -= torch.mean(new_template, dim=1, keepdim = True) # 1*1*3

                old_template = netE.vertices_init.cuda()
                whether_cross = torch.sum(torch.nn.functional.relu(-torch.sign(new_template[:,:,2])*torch.sign(old_template[:,:,2])))
                print('whether_cross:%f'%whether_cross)
                netE.vertices_init.data = new_template
                if whether_cross>0 and opt.cross: # only update when no point over depth
                    netE.vertices_init.data = old_template      
                if opt.swa: swa_modelE.module.vertices_init.data = netE.vertices_init.data
                opt.em_step = opt.em_step*0.99 # decay
                if opt.update_bn: update_bn(train_dataloader, netE)
        netE.train()

