import argparse
import os
import random
import math
import tqdm
import shutil
import imageio
import numpy as np
import trimesh

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
from networks import Discriminator, DiffRender, Landmark_Consistency, AttributeEncoder, weights_init, deep_copy
# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices


# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import Dataset
from utils import camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, Timer
from models.model import VGG19, CameraEncoder, ShapeEncoder, LightEncoder, TextureEncoder

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='./data/CUB_Data', help='path to dataset root dir')
parser.add_argument('--template_path', default='./template/sphere.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nk', type=int, default=5, help='size of kerner')
parser.add_argument('--nf', type=int, default=32, help='dim of unit channel')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='leaning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=1, type=int, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--multigpus', action='store_true', default=False, help='whether use multiple gpus mode')
parser.add_argument('--resume', action='store_true', default=True, help='whether resume ckpt')
parser.add_argument('--lambda_gan', type=float, default=0.0001, help='parameter')
parser.add_argument('--lambda_reg', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_data', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_ic', type=float, default=0.1, help='parameter')
parser.add_argument('--lambda_lc', type=float, default=0.001, help='parameter')
parser.add_argument('--azi_scope', type=float, default=360, help='parameter')
parser.add_argument('--elev_range', type=str, default="0~30", help='~ separated list of classes for the lsun data set')
parser.add_argument('--dist_range', type=str, default="2~6", help='~ separated list of classes for the lsun data set')

opt = parser.parse_args()
opt.outf = './log/'+ opt.name
print(opt)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#if torch.cuda.is_available():
#    cudnn.benchmark = True

train_dataset = Dataset(opt.dataroot, opt.imageSize, train=True)
test_dataset = Dataset(opt.dataroot, opt.imageSize, train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True, pin_memory=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, pin_memory=True, num_workers=int(opt.workers))



if __name__ == '__main__':
    # differentiable renderer
    diffRender = DiffRender(filename_obj=opt.template_path, image_size=opt.imageSize)

    # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
    netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                            azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                            nc=4, nk=opt.nk, nf=opt.nf)

    if opt.multigpus:
        netE = torch.nn.DataParallel(netE)
    netE = netE.cuda()

    # netL: for Landmark Consistency
    netL = Landmark_Consistency(num_landmarks=diffRender.num_faces, dim_feat=256, num_samples=64)
    if opt.multigpus:
        netL = torch.nn.DataParallel(netL)
    netL = netL.cuda()

    # netD: Discriminator
    netD = Discriminator(nc=4, nf=64)
    netD.apply(weights_init)
    if opt.multigpus:
        netD = torch.nn.DataParallel(netD)
    netD = netD.cuda()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerE = optim.Adam(list(netE.parameters()) + list(netL.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    # setup learning rate scheduler
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=opt.niter, eta_min=1e-6)
    schedulerE = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE, T_max=opt.niter, eta_min=1e-6)

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
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))
        else:
            start_iter = 0
            start_epoch = 0
            print("=> no checkpoint can be found")


    ori_dir = os.path.join(opt.outf, 'fid/ori')
    rec_dir = os.path.join(opt.outf, 'fid/rec')
    inter_dir = os.path.join(opt.outf, 'fid/inter')
    ckpt_dir = os.path.join(opt.outf, 'ckpts')
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    summary_writer = SummaryWriter(os.path.join(opt.outf + "/logs"))

    for epoch in range(start_epoch, opt.niter):
        for iter, data in enumerate(train_dataloader):
            with Timer("Elapsed time in update: %f"):
                ############################
                # (1) Update D network
                ###########################
                optimizerD.zero_grad()
                Xa = Variable(data['data']['images']).cuda()

                Ea = Variable(data['data']['edge']).cuda()
                batch_size = Xa.shape[0]

                # encode real
                Ae = netE(Xa)
                Xer, Ae = diffRender.render(**Ae)

                rand_a = torch.randperm(batch_size)
                rand_b = torch.randperm(batch_size)
                Aa = deep_copy(Ae, rand_a)
                Ab = deep_copy(Ae, rand_b)
                Ai = {}

                # linearly interpolate 3D attributes
                if opt.lambda_ic > 0.0:
                    # camera interpolation
                    alpha_camera = torch.empty((batch_size), dtype=torch.float32).uniform_(0.0, 1.0).cuda()
                    Ai['azimuths'] = - torch.empty((batch_size), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
                    Ai['elevations'] = alpha_camera * Aa['elevations'] + (1-alpha_camera) * Ab['elevations']
                    Ai['distances'] = alpha_camera * Aa['distances'] + (1-alpha_camera) * Ab['distances']

                    # shape interpolation
                    alpha_shape = torch.empty((batch_size, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).cuda()
                    Ai['vertices'] = alpha_shape * Aa['vertices'] + (1-alpha_shape) * Ab['vertices']
                    Ai['delta_vertices'] = alpha_shape * Aa['delta_vertices'] + (1-alpha_shape) * Ab['delta_vertices']

                    # texture interpolation
                    alpha_texture = torch.empty((batch_size, 1, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).cuda()
                    Ai['textures'] = alpha_texture * Aa['textures'] + (1.0 - alpha_texture) * Ab['textures']

                    # light interpolation
                    alpha_light = torch.empty((batch_size, 1), dtype=torch.float32).uniform_(0.0, 1.0).cuda()
                    Ai['lights'] = alpha_light * Aa['lights'] + (1.0 - alpha_light) * Ab['lights']
                else:
                    Ai = Ae

                # interpolated 3D attributes render images, and update Ai
                Xir, Ai = diffRender.render(**Ai)
                # predicted 3D attributes from above render images 
                Aire = netE(Xir.detach().clone())
                # render again to update predicted 3D Aire 
                _, Aire = diffRender.render(**Aire)

                # discriminate loss
                lossD_real = opt.lambda_gan * (-netD(Xa.detach().clone()).mean())
                lossD_fake = opt.lambda_gan * (netD(Xer.detach().clone()).mean() + \
                                            netD(Xir.detach().clone()).mean()) / 2.0

                # WGAN-GP
                lossD_gp = 10.0 * opt.lambda_gan * (compute_gradient_penalty(netD, Xa.data, Xer.data) + \
                                            compute_gradient_penalty(netD, Xa.data, Xir.data)) / 2.0

                lossD = lossD_real + lossD_fake + lossD_gp
                lossD.backward()
                optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                optimizerE.zero_grad()
                # GAN loss
                lossR_fake = opt.lambda_gan * (-netD(Xer).mean() - netD(Xir).mean()) / 2.0
                lossR_data = opt.lambda_data * diffRender.recon_data(Xer, Xa)

                # mesh regularization
                lossR_reg = opt.lambda_reg * (diffRender.calc_reg_loss(Ae) +  diffRender.calc_reg_loss(Ai)) / 2.0
                # lossR_flip = 0.002 * (diffRender.recon_flip(Ae) + diffRender.recon_flip(Ai))
                lossR_flip = 0.1 * (diffRender.recon_flip(Ae) + diffRender.recon_flip(Ai) + diffRender.recon_flip(Aire)) / 3.0

                # interpolated cycle consistency
                loss_cam, loss_shape, loss_texture, loss_light = diffRender.recon_att(Aire, deep_copy(Ai, detach=True))
                lossR_IC = opt.lambda_ic * (loss_cam + loss_shape + loss_texture + loss_light)

                # landmark consistency
                Le = Ae['faces_image']
                Li = Aire['faces_image']
                Fe = Ae['img_feats']
                Fi = Aire['img_feats']
                Ve = Ae['visiable_faces']
                Vi = Aire['visiable_faces']
                lossR_LC = opt.lambda_lc * (netL(Fe, Le, Ve).mean() + netL(Fi, Li, Vi).mean())
                
                # overall loss
                lossR = lossR_fake + lossR_reg + lossR_flip  + lossR_data + lossR_IC +  lossR_LC

                lossR.backward()
                optimizerE.step()

                print('Name: ', opt.outf)
                print('[%d/%d][%d/%d]\n'
                'LossD: %.4f lossD_real: %.4f lossD_fake: %.4f lossD_gp: %.4f\n'
                'lossR: %.4f lossR_fake: %.4f lossR_reg: %.4f lossR_data: %.4f '
                'lossR_IC: %.4f \n'
                    % (epoch, opt.niter, iter, len(train_dataloader),
                        lossD.item(), lossD_real.item(), lossD_fake.item(), lossD_gp.item(),
                        lossR.item(), lossR_fake.item(), lossR_reg.item(), lossR_data.item(),
                        lossR_IC.item()
                        )
                )
        schedulerD.step()
        schedulerE.step()

        if epoch % 1 == 0:
            summary_writer.add_scalar('Train/lr', schedulerE.get_last_lr()[0], epoch)
            summary_writer.add_scalar('Train/lossD', lossD.item(), epoch)
            summary_writer.add_scalar('Train/lossD_real', lossD_real.item(), epoch)
            summary_writer.add_scalar('Train/lossD_fake', lossD_fake.item(), epoch)
            summary_writer.add_scalar('Train/lossD_gp', lossD_gp.item(), epoch)
            summary_writer.add_scalar('Train/lossR', lossR.item(), epoch)
            summary_writer.add_scalar('Train/lossR_fake', lossR_fake.item(), epoch)
            summary_writer.add_scalar('Train/lossR_reg', lossR_reg.item(), epoch)
            summary_writer.add_scalar('Train/lossR_data', lossR_data.item(), epoch)
            summary_writer.add_scalar('Train/lossR_IC', lossR_IC.item(), epoch)
            summary_writer.add_scalar('Train/lossR_LC', lossR_LC.item(), epoch)
            summary_writer.add_scalar('Train/lossR_flip', lossR_flip.item(), epoch)

            num_images = Xa.shape[0]
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

            vutils.save_image(Ea.detach(),
                    '%s/current_edge.png' % (opt.outf), normalize=True)

            Ae = deep_copy(Ae, detach=True)
            vertices = Ae['vertices']
            faces = diffRender.faces
            textures = Ae['textures']
            azimuths = Ae['azimuths']
            elevations = Ae['elevations']
            distances = Ae['distances']
            lights = Ae['lights']

            texure_maps = to_pil_image(textures[0].detach().cpu())
            texure_maps.save('%s/current_mesh_recon.png' % (opt.outf), 'PNG')
            texure_maps.save('%s/epoch_%03d_mesh_recon.png' % (opt.outf, epoch), 'PNG')

            tri_mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy())
            tri_mesh.export('%s/current_mesh_recon.obj' % opt.outf)
            tri_mesh.export('%s/epoch_%03d_mesh_recon.obj' % (opt.outf, epoch))

            rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation.gif' % epoch)
            writer = imageio.get_writer(rotate_path, mode='I')
            loop = tqdm.tqdm(list(range(-int(opt.azi_scope/2), int(opt.azi_scope/2), 10)))
            loop.set_description('Drawing Dib_Renderer SphericalHarmonics')
            for delta_azimuth in loop:
                Ae['azimuths'] = - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(batch_size).cuda()
                predictions, _ = diffRender.render(**Ae)
                predictions = predictions[:, :3]
                image = vutils.make_grid(predictions)
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                writer.append_data(image)
            writer.close()
            current_rotate_path = os.path.join(opt.outf, 'current_rotation.gif')
            shutil.copyfile(rotate_path, current_rotate_path)

        if epoch % 2 == 0:
            epoch_name = os.path.join(ckpt_dir, 'epoch_%05d.pth' % epoch)
            latest_name = os.path.join(ckpt_dir, 'latest_ckpt.pth')
            state_dict = {
                'epoch': epoch,
                'netE': netE.state_dict(),
                'netD': netD.state_dict(),
                'optimizerE': optimizerE.state_dict(),
                'optimizerD': optimizerD.state_dict(),
            }
            torch.save(state_dict, latest_name)

        if epoch % 20 == 0 and epoch > 0:
            netE.eval()
            for i, data in tqdm.tqdm(enumerate(test_dataloader)):
                Xa = Variable(data['data']['images']).cuda()
                paths = data['data']['path']

                with torch.no_grad():
                    Ae = netE(Xa)
                    Xer, Ae = diffRender.render(**Ae)

                    Ai = deep_copy(Ae)
                    Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
                    Xir, Ai = diffRender.render(**Ai)

                    for i in range(len(paths)):
                        path = paths[i]
                        image_name = os.path.basename(path)
                        rec_path = os.path.join(rec_dir, image_name)
                        output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
                        output_Xer.save(rec_path, 'JPEG', quality=100)

                        inter_path = os.path.join(inter_dir, image_name)
                        output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
                        output_Xir.save(inter_path, 'JPEG', quality=100)

                        ori_path = os.path.join(ori_dir, image_name)
                        output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
                        output_Xa.save(ori_path, 'JPEG', quality=100)
            fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 32, True)
            print('Test recon fid: %0.2f' % fid_recon)
            summary_writer.add_scalar('Test/fid_recon', fid_recon, epoch)

            fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 32, True)
            print('Test rotation fid: %0.2f' % fid_inter)
            summary_writer.add_scalar('Test/fid_inter', fid_inter, epoch)
            netE.train()
