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
parser.add_argument('--outf', default='./log/Bird/SMR', help='folder to output images and model checkpoints')
parser.add_argument('--dataroot', default='./data/CUB_Data', help='path to dataset root dir')
parser.add_argument('--template_path', default='./template/sphere.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
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
print(opt)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WanING: You have a CUDA device, so you should probably run with --cuda")


train_dataset = Dataset(opt.dataroot, opt.imageSize, train=True)
test_dataset = Dataset(opt.dataroot, opt.imageSize, train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True, pin_memory=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, pin_memory=True, num_workers=int(opt.workers))

def deep_copy(att, index=None, detach=False):
    if index is None:
        index = torch.arange(att['distances'].shape[0]).cuda()

    copy_att = {}
    for key, value in att.items():
        copy_keys = ['azimuths', 'elevations', 'distances', 'vertices', 'delta_vertices', 'textures', 'lights']
        if key in copy_keys:
            if detach:
                copy_att[key] = value[index].clone().detach()
            else:
                copy_att[key] = value[index].clone()
    return copy_att


class DiffRender(object):
    def __init__(self, filename_obj, image_size):
        self.image_size = image_size
        # camera projection matrix
        camera_fovy = np.arctan(1.0 / 2.5) * 2
        self.cam_proj = generate_perspective_projection(camera_fovy, ratio=image_size/image_size)

        mesh = kal.io.obj.import_mesh(filename_obj, with_materials=True)
        # the sphere is usually too small (this is fine-tuned for the clock)

        # get vertices_init
        vertices = mesh.vertices
        vertices.requires_grad = False
        vertices_max = vertices.max(0, True)[0]
        vertices_min = vertices.min(0, True)[0]
        vertices = (vertices - vertices_min) / (vertices_max - vertices_min)
        vertices_init = vertices * 2.0 - 1.0 # (1, V, 3)

        # get face_uvs
        faces = mesh.faces
        uvs = mesh.uvs.unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        face_uvs.requires_grad = False

        self.num_faces = faces.shape[0]
        self.num_vertices = vertices_init.shape[0]
        face_size = 3

        # flip index
        # face_center = (vertices_init[0][faces[:, 0]] + vertices_init[0][faces[:, 1]] + vertices_init[0][faces[:, 2]]) / 3.0
        # face_center_flip = face_center.clone()
        # face_center_flip[:, 2] *= -1
        # self.flip_index = torch.cdist(face_center, face_center_flip).min(1)[1]
        
        # flip index
        vertex_center_flip = vertices_init.clone()
        vertex_center_flip[:, 2] *= -1
        self.flip_index = torch.cdist(vertices_init, vertex_center_flip).min(1)[1]

        ## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
        edges = torch.cat([faces[:,i:i+2] for i in range(face_size - 1)] +
                        [faces[:,[-1,0]]], dim=0)

        edges = torch.sort(edges, dim=1)[0]
        face_ids = torch.arange(self.num_faces, dtype=torch.long).repeat(face_size)
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        nb_edges = edges.shape[0]
        # edge to faces
        sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
        sorted_faces_ids = face_ids[order_edges_ids]
        # indices of first occurences of each key
        idx_first = torch.where(
            torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                                    (1,0), value=1))[0]
        num_faces_per_edge = idx_first[1:] - idx_first[:-1]
        # compute sub_idx (2nd axis indices to store the faces)
        offsets = torch.zeros(sorted_edges_ids.shape[0], dtype=torch.long)
        offsets[idx_first[1:]] = num_faces_per_edge
        sub_idx = (torch.arange(sorted_edges_ids.shape[0], dtype=torch.long) -
                torch.cumsum(offsets, dim=0))
        num_faces_per_edge = torch.cat([num_faces_per_edge,
                                    sorted_edges_ids.shape[0] - idx_first[-1:]],
                                    dim=0)
        max_sub_idx = 2
        edge2faces = torch.zeros((nb_edges, max_sub_idx), dtype=torch.long)
        edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids
        edge2faces = edge2faces

        ## Set up auxiliary laplacian matrix for the laplacian loss
        vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(self.num_vertices, faces)

        self.vertices_init = vertices_init
        self.faces = faces
        self.face_uvs = face_uvs
        self.edge2faces = edge2faces
        self.vertices_laplacian_matrix = vertices_laplacian_matrix

    def render(self, **attributes):
        azimuths = attributes['azimuths']
        elevations = attributes['elevations']
        distances = attributes['distances']
        batch_size = azimuths.shape[0]
        device = azimuths.device
        cam_proj = self.cam_proj.to(device)

        vertices = attributes['vertices']
        textures = attributes['textures']
        lights = attributes['lights']

        faces = self.faces.to(device)
        face_uvs = self.face_uvs.to(device)

        num_faces = faces.shape[0]

        object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        camera_up = torch.tensor([[0., 1., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        # camera_pos = torch.tensor([[0., 0., 4.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        camera_pos = camera_position_from_spherical_angles(distances, elevations, azimuths, degrees=True)
        cam_transform = generate_transformation_matrix(camera_pos, object_pos, camera_up)

        face_vertices_camera, face_vertices_image, face_normals = \
           prepare_vertices(vertices=vertices,
                faces=faces, camera_proj=cam_proj, camera_transform=cam_transform
            )

        face_normals_unit = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)
        face_attributes = [
            torch.ones((batch_size, num_faces, 3, 1), device=device),
            face_uvs.repeat(batch_size, 1, 1, 1),
            face_normals_unit
        ]

        image_features, soft_mask, face_idx = dibr_rasterization(
            self.image_size, self.image_size, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        # texture_coords, mask = image_features
        texmask, texcoord, imnormal = image_features

        texcolor = texture_mapping(texcoord, textures, mode='bilinear')
        coef = spherical_harmonic_lighting(imnormal, lights)
        image = texcolor * texmask * coef.unsqueeze(-1) + torch.ones_like(texcolor) * (1 - texmask)
        image = torch.clamp(image, 0, 1)
        render_img = image
        
        render_silhouttes = soft_mask[..., None]
        rgbs = torch.cat([render_img, render_silhouttes], axis=-1).permute(0, 3, 1, 2)

        attributes['face_normals'] = face_normals
        attributes['faces_image'] = face_vertices_image.mean(dim=2)
        attributes['visiable_faces'] = face_normals[:, :, -1] > 0.1
        return rgbs, attributes

    def recon_att(self, pred_att, target_att):
        def angle2xy(angle):
            angle = angle * math.pi / 180.0
            x = torch.cos(angle)
            y = torch.sin(angle)
            return torch.stack([x, y], 1)

        loss_azim = torch.pow(angle2xy(pred_att['azimuths']) -
                     angle2xy(target_att['azimuths']), 2).mean()
        loss_elev = torch.pow(angle2xy(pred_att['elevations']) -
                     angle2xy(target_att['elevations']), 2).mean()
        loss_dist = torch.pow(pred_att['distances'] - target_att['distances'], 2).mean()
        loss_cam = loss_azim + loss_elev + loss_dist

        loss_shape = torch.pow(pred_att['vertices'] - target_att['vertices'], 2).mean()
        loss_texture = torch.pow(pred_att['textures'] - target_att['textures'], 2).mean()
        loss_light = 0.1  * torch.pow(pred_att['lights'] - target_att['lights'], 2).mean()
        return loss_cam, loss_shape, loss_texture, loss_light

    def recon_data(self, pred_data, gt_data):
        image_weight = 0.1
        mask_weight = 1.

        pred_img = pred_data[:, :3]
        pred_mask = pred_data[:, 3]
        gt_img = gt_data[:, :3]
        gt_mask = gt_data[:, 3]
        loss_image = torch.mean(torch.abs(pred_img - gt_img))
        loss_mask = kal.metrics.render.mask_iou(pred_mask, gt_mask)

        loss_data = image_weight * loss_image + mask_weight * loss_mask
        return loss_data

    def recon_flip(self, att):
        Na = att['delta_vertices']
        Nf = Na.index_select(1, self.flip_index.to(Na.device))
        Nf[..., 2] *= -1

        loss_norm = (Na - Nf).norm(dim=2).mean()
        return loss_norm

    def calc_reg_loss(self, att):
        laplacian_weight = 0.1
        flat_weight = 0.001

        # laplacian loss
        delta_vertices = att['delta_vertices']
        device = delta_vertices.device

        vertices_laplacian_matrix = self.vertices_laplacian_matrix.to(device)
        edge2faces = self.edge2faces.to(device)
        face_normals = att['face_normals']
        nb_vertices = delta_vertices.shape[1]
        
        delta_vertices_laplacian = torch.matmul(vertices_laplacian_matrix, delta_vertices)
        loss_laplacian = torch.mean(delta_vertices_laplacian ** 2) * nb_vertices * 3
        # flat loss
        mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
        mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
        faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
        loss_flat = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]

        loss_reg = laplacian_weight * loss_laplacian + flat_weight * loss_flat
        return loss_reg


# network of landmark consistency
class Landmark_Consistency(nn.Module):
    def __init__(self, num_landmarks, dim_feat, num_samples):
        super(Landmark_Consistency, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_samples = num_samples

        n_features = dim_feat
        self.classifier = nn.Sequential(
            nn.Conv1d(n_features, 1024, 1, 1, 0), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Conv1d(1024, self.num_landmarks, 1, 1, 0)
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, img_feat, landmark_2d, visiable):
        batch_size = landmark_2d.shape[0]
        grid_x = landmark_2d.unsqueeze(1)  # (N, 1, V, 2)

        feat_sampled = F.grid_sample(img_feat, grid_x, mode='bilinear', padding_mode='zeros')  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feature_agg = feat_sampled.transpose(1, 2)   # (N, F, V)
        # feature_agg = torch.cat([feat_sampled, landmark_2d], dim=2).transpose(1, 2) # (B, F, V)

        # select feature
        select_index = torch.randperm(self.num_landmarks)[:self.num_samples].cuda()
        feature_agg = feature_agg.index_select(2, select_index) # (B, F, 64)
        logits = self.classifier(feature_agg) # (B, num_landmarks, 64)
        logits = logits.transpose(1, 2).reshape(-1, self.num_landmarks) # (B*64, num_landmarks)

        labels = torch.arange(self.num_landmarks)[None].repeat(batch_size, 1).cuda() # (B, V)
        labels = labels.index_select(1, select_index).view(-1) # (B*64,)

        visiable = visiable.index_select(1, select_index).view(-1).float()
        loss = (self.cross_entropy(logits, labels) * visiable).sum() / visiable.sum()
        return loss

class AttributeEncoder(nn.Module):
    def __init__(self, num_vertices, vertices_init, azi_scope, elev_range, dist_range, nc, nf, nk):
        super(AttributeEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init

        self.camera_enc = CameraEncoder(nc=nc, nk=nk, azi_scope=azi_scope, elev_range=elev_range, dist_range=dist_range)
        self.shape_enc = ShapeEncoder(nc=nc, nk=nk, num_vertices=self.num_vertices)
        self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        self.light_enc = LightEncoder(nc=nc, nk=nk)
        # self.feat_enc = FeatEncoder(nc=4, nf=32)
        self.feat_enc = VGG19()

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        input_img = x

        # cameras
        cameras = self.camera_enc(input_img)
        azimuths, elevations, distances = cameras

        # vertex
        delta_vertices = self.shape_enc(input_img)
        vertices = self.vertices_init[None].to(device) + delta_vertices

        # textures
        textures = self.texture_enc(input_img)
        lights = self.light_enc(input_img)

        # image feat
        with torch.no_grad():
            self.feat_enc.eval()
            img_feats = self.feat_enc(input_img)

        # others
        attributes = {
        'azimuths': azimuths,
        'elevations': elevations,
        'distances': distances,
        'vertices': vertices,
        'delta_vertices': delta_vertices,
        'textures': textures,
        'lights': lights,
        'img_feats': img_feats
        }
        return attributes


class Discriminator(nn.Module):
    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(nf * 4, nf * 8, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(nf * 8, nf * 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(nf * 16, nf * 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 8, 1, 1, 1, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input).mean([2, 3])
        return output


# custom weights initialization called on netE and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
