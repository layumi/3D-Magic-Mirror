import torch
import numpy as np
import math
import time
import torch
import random
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

SMOOTH = 1e-6

import pathlib
import warnings
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont


def _base_face_clocks(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = x2 * y3 - x3 * y2
    b = x3 * y1 - x1 * y3
    c = x1 * y2 - x2 * y1
    clocks = 0.5*(a + b + c)

    return clocks

def face_clocks(vertices, faces):
    """Compute the areas of each face of triangle meshes.
    Args:
        vertices (torch.Tensor):
            The vertices of the meshes,
            of shape :math:`(\\text{batch_size}, \\text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            the faces of the meshes, of shape :math:`(\\text{num_faces}, 3)`.
    Returns:
        (torch.Tensor):
            the face areas of same type as vertices and of shape
            :math:`(\\text{batch_size}, \\text{num_faces})`.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("face_areas is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))

    clocks = _base_face_clocks(face_v_0, face_v_1, face_v_2)
    return clocks.squeeze(-1)

def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 6,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        range (tuple. optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``value_range``
                instead.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if "range" in kwargs.keys():
        warnings.warn(
            "The parameter 'range' is deprecated since 0.12 and will be removed in 0.14. "
            "Please use 'value_range' instead."
        )
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    assert isinstance(tensor, torch.Tensor)
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding-4, width * xmaps + padding-4), pad_value)
    grid = torch.nn.functional.pad(grid, (4,4,4,4), value =1.0)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def white(Ae0):
    v = Ae0['vertices']
    Ae0['vertices'] -= torch.mean(v, dim=1, keepdim = True).repeat(1, v.shape[1], 1)
    va = Ae0['delta_vertices']
    Ae0['delta_vertices'] -= torch.mean(va, dim=1, keepdim = True).repeat(1, va.shape[1], 1)
    return Ae0

def angle2xy(angle):
    angle = angle * math.pi / 180.0
    x = torch.cos(angle)
    y = torch.sin(angle)
    return torch.stack([x, y], 1)

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = torch.logical_and(outputs, labels).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.logical_or(outputs, labels).sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    #print('Mean IoU: %.2f'% torch.mean(iou))    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
def save_mesh(obj_mesh_name, v, faces, vt=None):
    with open(obj_mesh_name, 'w') as fp:
        for i in range(v.shape[0]):
            fp.write( 'v %f %f %f\n' % ( v[i,0], v[i,1], v[i,2]))
        if not vt is None:
            for i in range(vt.shape[0]):
                fp.write( 'vt %f %f\n' %  (vt[i,0], vt[i,1]) )
        for f in faces:  # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

def mask(gt_data):
    gt_img = gt_data[:, :3]
    gt_mask = gt_data[:, 3]
    gt_img = gt_img * gt_mask.unsqueeze(1) + torch.ones_like(gt_img) * (1 - gt_mask.unsqueeze(1))
    return gt_img

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def ChannelShuffle(img):
    # rgb -> rbg,
    rand = random.uniform(0, 1)
    if rand<0.2:
        inv_idx = [0,2,1,3]
    elif rand<0.4:
        inv_idx = [1,0,2,3]
    elif rand<0.6:
        inv_idx = [1,2,0,3]
    elif rand<0.8:
        inv_idx = [2,0,1,3]
    else:
        inv_idx = [2,1,0,3]

    inv_idx = torch.LongTensor(inv_idx).long().cuda()  # N x C x H x W
    img_shuffle = img.index_select(1,inv_idx)
    return img_shuffle


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def camera_position_from_spherical_angles(dist, elev, azim, degrees=True):
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.
    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.
    The vectors are broadcast against each other so they all have shape (N, 1).
    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    return camera_position.reshape(-1, 3)


def generate_transformation_matrix(camera_position, look_at, camera_up_direction):
    r"""Generate transformation matrix for given camera parameters.

    Formula is :math:`\text{P_cam} = \text{P_world} * {\text{transformation_mtx}`,
    with :math:`\text{P_world}` being the points coordinates padded with 1.

    Args:
        camera_position (torch.FloatTensor):
            camera positions of shape :math:`(\text{batch_size}, 3)`,
            it means where your cameras are
        look_at (torch.FloatTensor):
            where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
        camera_up_direction (torch.FloatTensor):
            camera up directions of shape :math:`(\text{batch_size}, 3)`,
            it means what are your camera up directions, generally [0, 1, 0]

    Returns:
        (torch.FloatTensor):
            The camera transformation matrix of shape :math:`(\text{batch_size, 4, 3)`.
    """
    z_axis = (camera_position - look_at)
    z_axis = z_axis / z_axis.norm(dim=1, keepdim=True)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis = x_axis / x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = (-camera_position.unsqueeze(1) @ rot_part)
    return torch.cat([rot_part, trans_part], dim=1)


def compute_gradient_penalty_list(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    for iter, _ in enumerate(d_interpolates):
        fake = Variable(Tensor(torch.ones_like(d_interpolates[iter])), requires_grad=False)
        gradients = autograd.grad(
        outputs=d_interpolates[iter],
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        )[0]
        gradients = gradients.contiguous().view(gradients.size(0), -1)
        if iter==0:
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        else:
            gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(torch.ones_like(d_interpolates)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



# def calc_fid():
#     for i, data in tqdm.tqdm(enumerate(test_dataloader)):
#         Xa = Variable(data['data']['images']).cuda()
#         paths = data['data']['path']

#         with torch.no_grad():
#             Ae = netE(Xa)
#             Xer, Ae = diffRender.render(**Ae)

#             Ai = deep_copy(Ae)
#             Ai['azimuths'] = - torch.empty((Xa.shape[0]), dtype=torch.float32).uniform_(-opt.azi_scope/2, opt.azi_scope/2).cuda()
#             Xir, Ai = diffRender.render(**Ai)

#             for i in range(len(paths)):
#                 path = paths[i]
#                 image_name = os.path.basename(path)
#                 rec_path = os.path.join(rec_dir, image_name)
#                 output_Xer = to_pil_image(Xer[i, :3].detach().cpu())
#                 output_Xer.save(rec_path, 'JPEG', quality=100)

#                 inter_path = os.path.join(inter_dir, image_name)
#                 output_Xir = to_pil_image(Xir[i, :3].detach().cpu())
#                 output_Xir.save(inter_path, 'JPEG', quality=100)

#                 ori_path = os.path.join(ori_dir, image_name)
#                 output_Xa = to_pil_image(Xa[i, :3].detach().cpu())
#                 output_Xa.save(ori_path, 'JPEG', quality=100)
#     fid_recon = calculate_fid_given_paths([ori_dir, rec_dir], 32, True)
#     print('Test recon fid: %0.2f' % fid_recon)
#     summary_writer.add_scalar('Test/fid_recon', fid_recon, epoch)

#     fid_inter = calculate_fid_given_paths([ori_dir, inter_dir], 32, True)
#     print('Test rotation fid: %0.2f' % fid_inter)
#     summary_writer.add_scalar('Test/fid_inter', fid_inter, epoch)
