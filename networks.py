import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import kaolin as kal
import math
import copy
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices
#from models.model import TextureEncoder
from network.model_res import VGG19, TextureEncoder, BackgroundEncoder, CameraEncoder, ShapeEncoder, LightEncoder
from network.utils import weights_init, weights_init_classifier
from smr_utils import face_clocks, camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, Timer
from fid_score import calculate_fid_given_paths
from pytorch3d.loss import chamfer_distance
#import sys
#sys.path.append('./ROMP/romp/lib/')
#from ROMP.romp.predict.image_simple import Image_processor

class MS_Discriminator(nn.Module):
    def __init__(self, nc = 4, nf = 32, use_bias = True):
        super(MS_Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        self.num_scales = 3
        self.use_bias = use_bias
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        for _ in range(self.num_scales):
            Dis = self._make_net(nc, nf)
            self.cnns.append(Dis)

    def _make_net(self, nc, nf):
        cnn_x = nn.Sequential(
            nn.Conv2d(nc, nf//2, 1, 1, 0, bias=self.use_bias), # 4 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf//2, nf//2, 3, 1, 1, bias=self.use_bias), # 16 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(nf//2, nf , 3, 2, 1, bias=self.use_bias), # 16 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf , 3, 1, 1, bias=self.use_bias), # 32 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(nf , nf , 3, 2, 1, bias=self.use_bias), # 32 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf , 3, 1, 1, bias=self.use_bias), # 32 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=self.use_bias), # 32 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*2 , 3, 1, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8 
            nn.Conv2d(nf*2, nf * 2, 3, 2, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 2, nf * 2, 1, 1, 0, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, 1, 1, 1, 0, bias=self.use_bias) # 64-> 1
        )
        cnn_x.apply(weights_init)
        cnn_x[-1].apply(weights_init_classifier)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.contiguous().view(batch_size, -1).sum(1)
        return reg

class Discriminator(nn.Module):
    def __init__(self, nc = 3, nf = 16, use_bias = False):
        super(Discriminator, self).__init__()
        self.use_bias = use_bias
        self.main = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(nc, nf, 1, 1, 0, bias=self.use_bias), # 3 -> 16  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=self.use_bias), # 16 -> 16  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=self.use_bias), # 16 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(nf*2, nf*2, 3, 1, 1, bias=self.use_bias), # 32 -> 32  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 3, 3, 2, 1, bias=self.use_bias), # 32 -> 48
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(nf*3, nf*3, 3, 1, 1, bias=self.use_bias), # 32 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 3, nf * 4, 3, 2, 1, bias=self.use_bias), # 48 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            # 4 ->2
            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=self.use_bias), # 64 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 3, 3, 2, 1, bias=self.use_bias), # 64 -> 48
            nn.LeakyReLU(0.2, inplace=True),
            # if (128*64)-> 2*1
            nn.Conv2d(nf * 3, nf * 2, 1, 1, 0, bias=self.use_bias), # 48 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, 1, 1, 1, 0, bias=self.use_bias) # 32->1
        )
        self.main.apply(weights_init)
        self.main[-1].apply(weights_init_classifier)
    def forward(self, input):
        outputs = self.main(input).mean([2, 3])
        return outputs

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.contiguous().view(batch_size, -1).sum(1)
        return reg

def deep_copy(att, index=None, detach=False):
    if index is None:
        index = torch.arange(att['distances'].shape[0], device='cuda') # sequential order

    copy_att = {}
    for key, value in att.items():
        copy_keys = ['azimuths', 'bg', 'biases', 'elevations', 'distances', 'vertices', 'delta_vertices', 'textures', 'lights']
        if key in copy_keys:
            if value is None:
                copy_att[key] = None
                continue
            if detach:
                copy_att[key] = value[index].clone().detach()
            else:
                copy_att[key] = value[index].clone()
    return copy_att


class DiffRender(object):
    def __init__(self, mesh_name, image_size, ratio=1, init_ellipsoid = 1, image_weight=0.1, lambda_lpl = 0.1, lambda_flat = 0.001):
        self.image_size = image_size
        self.image_weight = image_weight
        self.lambda_lpl = lambda_lpl
        self.lambda_flat = lambda_flat
        self.ratio = ratio
        # camera projection matrix
        camera_fovy = np.arctan(1.0 / 2.5) * 2
        # here ratio=width/height
        self.cam_proj = generate_perspective_projection(camera_fovy, ratio=1/ratio)

        mesh = kal.io.obj.import_mesh(mesh_name, with_materials=True)
        print('Vertices Number:', mesh.vertices.shape[0]) #642
        print('Faces Number:', mesh.faces.shape)  #1280
        # https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dibr_tutorial.ipynb
        # get vertices_init
        vertices = mesh.vertices
        vertices.requires_grad = False
        vertices_max = vertices.max(0, True)[0]
        vertices_min = vertices.min(0, True)[0]
        vertices = (vertices - vertices_min) / (vertices_max - vertices_min)
        vertices_init = vertices * 2.0 - 1.0 # (V, 3)
        # ellipsoid for better initial gradients
        if not init_ellipsoid == -1:
            vertices_init[:,2] = vertices_init[:,2] / 2 # depth = 1/2 * height. z axis is different from x axis. 
            if init_ellipsoid != 1:
                print('using the ellipsoid template')
                vertices_init[:,0] = vertices_init[:,0] / init_ellipsoid  # width = 1/2 * height
                vertices_init[:,2] = vertices_init[:,2] / init_ellipsoid  # depth = 1/4 * height
        vertices_init *= 0.9 # scale a little small
        # get face_uvs
        faces = mesh.faces
        uvs = mesh.uvs.unsqueeze(0)
        self.uvs = mesh.uvs
        print(self.uvs.shape)
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
        self.edges = edges
        print('Unique Edge Number: %d'%nb_edges)
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

        ## Set up auxiliary laplacian matrix for the laplacian loss
        vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(self.num_vertices, faces)

        self.vertices_init = vertices_init
        self.sign_init = torch.sign(vertices_init[:,2]).cuda()
        self.faces = faces
        self.face_uvs = face_uvs
        self.edge2faces = edge2faces
        self.vertices_laplacian_matrix = vertices_laplacian_matrix

    def render(self, no_mask=False, **attributes):
        azimuths = attributes['azimuths']
        elevations = attributes['elevations']
        distances = attributes['distances']
        biases = attributes['biases']
        bg = attributes['bg']
        batch_size = azimuths.shape[0]
        device = azimuths.device
        cam_proj = self.cam_proj.to(device)

        vertices = attributes['vertices']
        textures = attributes['textures']
        lights = attributes['lights']

        faces = self.faces.to(device)
        face_uvs = self.face_uvs.to(device)
        #face_areas = kal.ops.mesh.face_areas(vertices, faces)

        num_faces = faces.shape[0]
        #object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        object_pos = torch.cat((biases, torch.zeros(batch_size, 1, device=device)), dim=1) # N*2 + 0 ->N*3
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
            round(self.ratio*self.image_size), self.image_size, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        # texture_coords, mask = image_features
        texmask, texcoord, imnormal = image_features

        texcolor = texture_mapping(texcoord, textures, mode='bilinear')
        coef = spherical_harmonic_lighting(imnormal, lights)
        if no_mask:
            bg= bg.permute(0, 2, 3, 1)
            #print(texcolor.shape, texmask.shape, bg.shape)
            image = texcolor * texmask +  bg * (1 - texmask)
            image *= coef.unsqueeze(-1)
        else:
            image = texcolor * texmask * coef.unsqueeze(-1) + torch.ones_like(texcolor) * (1 - texmask)
        render_img = torch.clamp(image, 0, 1)
        
        render_silhouttes = soft_mask[..., None]
        rgbs = torch.cat([render_img, render_silhouttes], axis=-1).permute(0, 3, 1, 2)

        attributes['face_normals'] = face_normals
        attributes['imnormal'] = imnormal # visualize only. not train
        #attributes['face_areas'] = face_areas
        #attributes['faces_image'] = face_vertices_image.mean(dim=2)
        #attributes['visible_faces'] = face_normals[:, :, -1] > 0.1
        return rgbs, attributes

    def recon_att(self, pred_att, target_att, L1 = False, chamfer = False, azim=1):
        def angle2xy(angle):
            angle = angle * math.pi / 180.0
            x = torch.cos(angle)
            y = torch.sin(angle)
            return torch.stack([x, y], 1)

        if L1:
            loss_azim = torch.abs(angle2xy(pred_att['azimuths']) -
                     angle2xy(target_att['azimuths'])).mean()
            loss_elev = torch.abs(angle2xy(pred_att['elevations']) -
                     angle2xy(target_att['elevations'])).mean()
            loss_dist = torch.abs(pred_att['distances'] - target_att['distances']).mean()
            loss_bias = torch.abs(pred_att['biases'] - target_att['biases']).mean()
            loss_cam = azim * loss_azim + loss_elev + loss_dist
            if chamfer:
                loss_shape, _  = chamfer_distance(pred_att['vertices'], target_att['vertices'])
            else:
                loss_shape = torch.abs(pred_att['vertices'] - target_att['vertices']).mean()
            loss_texture = torch.abs(pred_att['textures'] - target_att['textures']).mean()
            loss_light = 0.1  * torch.abs(pred_att['lights'] - target_att['lights']).mean()
        else:
            loss_azim = torch.pow(angle2xy(pred_att['azimuths']) -
                     angle2xy(target_att['azimuths']), 2).mean()
            loss_elev = torch.pow(angle2xy(pred_att['elevations']) -
                     angle2xy(target_att['elevations']), 2).mean()
            loss_dist = torch.pow(pred_att['distances'] - target_att['distances'], 2).mean()
            loss_bias = torch.pow(pred_att['biases'] - target_att['biases'], 2).mean()
            loss_cam = azim * loss_azim + loss_elev + loss_dist
            if chamfer:
                loss_shape, _  = chamfer_distance(pred_att['vertices'], target_att['vertices'])
            else:
                loss_shape = torch.pow(pred_att['vertices'] - target_att['vertices'], 2).mean()
            loss_texture = torch.pow(pred_att['textures'] - target_att['textures'], 2).mean()
            loss_light = 0.1  * torch.pow(pred_att['lights'] - target_att['lights'], 2).mean()

        return loss_cam, loss_shape, loss_texture, loss_light, loss_bias

    def recon_data(self, pred_data, gt_data, no_mask=False, contour=0):
        image_weight = self.image_weight
        mask_weight = 1.

        pred_img = pred_data[:, :3]
        pred_mask = pred_data[:, 3]
        gt_img = gt_data[:, :3]
        gt_mask = gt_data[:, 3]
        #if no_mask:
        #print(gt_img.shape, gt_mask.shape)
        gt_img = gt_img * gt_mask.unsqueeze(1) + torch.ones_like(gt_img) * (1 - gt_mask.unsqueeze(1))
        pred_img = pred_img * gt_mask.unsqueeze(1) + torch.ones_like(pred_img) * (1 - gt_mask.unsqueeze(1))
        loss_image = torch.mean(torch.abs(pred_img - gt_img))
        loss_mask = kal.metrics.render.mask_iou(pred_mask, gt_mask)
       
        if contour>0: 
            n, h, w = gt_mask.shape 
            gt_contour = gt_mask.unsqueeze(1) - F.interpolate(F.interpolate(gt_mask.unsqueeze(1), size=(h//4,w//4)), size=(h,w))
            pred_contour = pred_mask.unsqueeze(1) - F.interpolate(F.interpolate(pred_mask.unsqueeze(1),  size=(h//4,w//4)), size=(h,w))
            gt_contour, pred_contour = torch.abs(gt_contour), torch.abs(pred_contour)
            #loss_contour = torch.mean(torch.abs(pred_contour - gt_contour))
            loss_contour = torch.mean((pred_contour - gt_contour)**2)
            loss_mask += loss_contour*contour
            print('loss_contour: %f'%loss_contour)

        loss_data = image_weight * loss_image + mask_weight * loss_mask
        return loss_data

    def recon_flip(self, att, L1):
        Na = att['delta_vertices']
        Nf = Na.index_select(1, self.flip_index.to(Na.device)) #.detach()
        Nf[..., 2] *= -1
        # control the symmetry along the z axis.
        if L1: # encourage hand to move out. 
            loss_norm = torch.abs(Na - Nf)
        else:
            loss_norm = (Na - Nf).norm(dim=2)
        # print(Na.shape, loss_norm.shape)
        # ignore the wrong sign. swapping edge
        # Finding swapped points. Same sign, mask_a = 1;  otherwise mask_a=0
        mask_a = torch.nn.functional.relu(torch.sign(Na[:,:,2]) * self.sign_init)
        mask_f = mask_a.index_select(1, self.flip_index.to(Na.device))
        # Finding swapped point pairs.  mask = 0; otherwise mask=1
        mask = torch.logical_and(mask_a.detach(), mask_f.detach())
        #print(mask)
        loss_norm = loss_norm* mask_f 
        return torch.mean(loss_norm)

    def calc_reg_loss(self, att):
        laplacian_weight = self.lambda_lpl #0.1
        flat_weight = self.lambda_flat #0.001

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

        # face normal loss # inner face, which cosine < -0.5
        #clocks = face_clocks(att['vertices'], self.faces.cuda())
        #loss_clock = torch.nn.functional.relu(clocks)**2
        #print(torch.sum(loss_clock>1e-6, dim=1))
        #loss_clock = torch.mean(torch.sum(loss_clock, dim =1))
        #print('loss_clock:%f'%loss_clock)
        #loss_flat += loss_clock*10
        #loss_invisible = torch.mean(torch.nn.functional.relu(-faces_cos-0.5)**2)* edge2faces.shape[0]
        #loss_flat += loss_invisible 

        # area loss
        #face_areas = att['face_areas']
        #mean_areas = torch.mean(face_areas, dim=1, keepdim=True)
        #bias_areas = face_areas-mean_areas
        #loss_areas = torch.mean(torch.norm(bias_areas, p=2, dim=1))
        #print('loss_invisible:%.2f, loss_area;%2f'%(loss_invisible, loss_areas))
        #print('loss_invisible:%.2f'%loss_invisible)
        loss_reg = laplacian_weight * loss_laplacian + flat_weight * loss_flat #+ 0.1*loss_areas
        return loss_reg

    def calc_reg_edge(self, pred): # pred is  att['vertices'] 
        #batchsize = pred.shape[0]
        edge_weight = 0.1
        edge_length = torch.norm( (pred[:, self.edges[:, 0]] - pred[:, self.edges[:, 1]]), p =2, dim=2)
        mean_length = torch.mean(edge_length, dim=1, keepdim=True)
        bias_length = edge_length-mean_length
        loss_edge = edge_weight*torch.mean(torch.norm(bias_length, p=2, dim=1) ) # have to be mse , otherwise sparse L1 case.

        return loss_edge

    def calc_reg_depth(self, pred): # pred is  att['vertices']
        # L2 regularization on depth 
        loss_depth = torch.mean(pred[:,:,2]**2 ) 
        return loss_depth

    def calc_reg_depthR(self, pred, temp = 2, eps = 0.001): # exp Circle. pred is  att['vertices']
        # L2 regularization on depth w R
        x = pred[:,:,0].detach()
        y = pred[:,:,1].detach()
        #loss_depth = pred[:,:,2]**2 * torch.exp( 2*(x**2 + (y/self.ratio)**2))
        # keep sign
        loss_depth = (self.sign_init>=0)* (pred[:,:,2]-eps)**2 * torch.exp( temp*(x**2 + (y/self.ratio)**2)) + (self.sign_init<0)*(pred[:,:,2]+eps)**2 * torch.exp( temp*(x**2 + (y/self.ratio)**2)) # prevent all to zero. positive to 0.001, negative to -0.001.
        return torch.mean(loss_depth)

    def calc_reg_depthC(self, pred, eps = 0.001): # Circle . pred is  att['vertices']
        # L2 regularization on depth w R
        x = pred[:,:,0].detach()
        y = pred[:,:,1].detach()
        #z = pred[:,:,2].detach()
        #loss_depth = pred[:,:,2]**2 * (x**2 + (y/self.ratio)**2)
        # keep sign
        loss_depth = (self.sign_init>=0)* (pred[:,:,2]-eps)**2 * (x**2 + (y/self.ratio)**2) + (self.sign_init<0)* (pred[:,:,2]+eps)**2 * (x**2 + (y/self.ratio)**2)
        return torch.mean(loss_depth)

    def calc_reg_deform(self, pred): # pred is  att['delta_vertices'], x,y,z. B*N*3
        batchsize = pred.shape[0] 
        pred = pred.reshape(-1, pred.size(2)) # ((B*N)*3)
        norm = torch.norm(pred, p=2, dim=1).reshape(batchsize, -1) # B * N
        return torch.mean(norm)
        # >0.25 will be penalized more
        #return torch.mean(  (2*torch.mean(norm, dim=1) ) **2 )

# network of landmark consistency
class Landmark_Consistency(nn.Module):
    def __init__(self, num_landmarks=1280, dim_feat=256, num_samples=64):
        super(Landmark_Consistency, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_samples = num_samples

        self.classifier = nn.Sequential(
            nn.Conv1d(dim_feat, 1024, 1, 1, 0), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Conv1d(1024, self.num_landmarks, 1, 1, 0)
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.classifier.apply(weights_init)
        self.classifier[-1].apply(weights_init_classifier)

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
    def __init__(self, num_vertices=642, vertices_init=None, azi_scope=360, elev_range='0-30', dist_range='2-6', nc=4, nf=32, nk=5, ratio=1, makeup=False, bg = False, pretraint = 'res34', pretrainc='hr18', pretrains='hr18', droprate=0.0, romp=False, coordconv=False, norm = 'bn', lpl = None, nolpl=False, inv = 0):
        super(AttributeEncoder, self).__init__()
        self.num_vertices = num_vertices # 642
        self.vertices_init = vertices_init[None].cuda() # (1, V, 3) in [-1,1]

        if isinstance(droprate, str):
            # camera, light use one dropout. 
            # texture, background use one dropout.
            droprates = droprate.split(',')
            droprate_c, droprate_s, droprate_t = float(droprates[0]), float(droprates[1]), float(droprates[2])
        else: 
            droprate_c, droprate_s, droprate_t = 0.2, 0.2, 0.2 
        print('DropRate: %.2f, %.2f, %.2f'%(droprate_c, droprate_s, droprate_t/2) )
        self.camera_enc = CameraEncoder(nc=nc, nk=nk, azi_scope=azi_scope, elev_range=elev_range, dist_range=dist_range, droprate = droprate_c, coordconv=coordconv, norm = norm, ratio = ratio, pretrain = pretrainc, nolpl = nolpl)
        self.shape_enc = ShapeEncoder(nc=nc, nk=nk, num_vertices=self.num_vertices, pretrain = pretrains, droprate = droprate_s, coordconv=coordconv, norm=norm, nolpl = nolpl)
        self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices, pretrain = pretraint, ratio = ratio, makeup = makeup, droprate = droprate_t, coordconv=coordconv, norm=norm)
        self.light_enc = LightEncoder(nc=nc, nk=nk, droprate=droprate_c, coordconv=coordconv, norm=norm)
        self.bg = bg
        if self.bg:
            self.bg_enc = BackgroundEncoder(nc=nc, droprate = droprate_t, coordconv=coordconv)
        # self.feat_enc = FeatEncoder(nc=4, nf=32)
        self.romp = romp
        if self.romp:
            self.romp_enc = Image_processor()
        self.lpl = lpl # 642*642
        self.inv = inv
        if self.inv>0:
            M = torch.linalg.inv((torch.eye(lpl.size(0)) + inv*lpl)).cuda() # M = (I+3L)^(-1)
            self.M = M*M
        #self.feat_enc = VGG19()
        #self.feat_enc.eval()

    def forward(self, input_img, need_feats=False, img_pth = None, train_shape = 0):
        if type(input_img) == dict: # for swa update_bn function
            input_img = input_img['data']['images'].cuda().detach()

        device = input_img.device
        batch_size = input_img.shape[0]

        def vertice_hook(grad):
            # https://rgl.epfl.ch/publications/Nicolet2021Large
            grad_inv = grad.clone()
            grad_inv = torch.permute(grad_inv, (0, 2, 1))
            grad_inv = torch.matmul(grad_inv, self.M) # M = (I+3L)^(-2)
            grad = torch.permute(grad_inv, (0, 2, 1))
            return grad

        # camera + vertex 
        if train_shape == 1 or train_shape ==4 or train_shape ==5: 
            print("Fix Shape Encoder", end=', ')
            shape_enc_copy = copy.deepcopy(self.shape_enc)
            with torch.no_grad():
                delta_vertices = shape_enc_copy(input_img, template = self.vertices_init, lpl = self.lpl)
        else:
            print("Train Shape Encoder", end=', ') 
            delta_vertices = self.shape_enc(input_img, template = self.vertices_init, lpl = self.lpl) # 32 x 642x 3
            if self.inv >0 and delta_vertices.requires_grad:
                delta_vertices.register_hook(vertice_hook)

        if train_shape == 2 or train_shape == 3 or train_shape == 4:
            print("Fix Camera Encoder", end=', ')
            camera_enc_copy = copy.deepcopy(self.camera_enc)
            with torch.no_grad():
                cameras = camera_enc_copy(input_img, template = self.vertices_init)
            azimuths, elevations, distances, biases = cameras # 32, 32, 32
        else:
            print("Train Camera Encoder", end=', ')
            cameras = self.camera_enc(input_img, template = self.vertices_init)
            azimuths, elevations, distances, biases = cameras # 32, 32, 32

        if train_shape == 3 or train_shape == 5:
            print("Fix Texture Encoder")
            texture_enc_copy = copy.deepcopy(self.texture_enc)
            light_enc_copy = copy.deepcopy(self.light_enc)
            with torch.no_grad():
                textures = texture_enc_copy(input_img) # 32x3x512x256
                lights = light_enc_copy(input_img) # 32x9
        else:
            print("Train Texture Encoder")
            textures = self.texture_enc(input_img) # 32x3x512x256
            lights = self.light_enc(input_img) # 32x9

        if self.romp and img_path is not None:
            vertices = self.romp_enc.run(file_list=img_pth).to(device) 
            print(vertices.shape, delta_vertices.shape)
            vertices += delta_vertices # 32 x 6890 x 3
        else:
            vertices = self.vertices_init + delta_vertices

        # background
        if self.bg:
            background = self.bg_enc(input_img)
        else:
            background = None

        # image feat
        #if need_feats:
        #    with torch.no_grad():
        #        img_feats = self.feat_enc(input_img) # 32x256x32x32
        #else:
        img_feats = None #
        # others
        attributes = {
        'azimuths': azimuths,
        'elevations': elevations,
        'distances': distances,
        'biases': biases,
        'vertices': vertices,
        'delta_vertices': delta_vertices,
        'textures': textures,
        'lights': lights,
        'img_feats': img_feats,
        'bg': background,
        }
        return attributes

