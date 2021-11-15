import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import math

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        B, C, H, W = X.shape
        X = normalize_batch(X[:, :3])
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [F.interpolate(h_relu3, [H//2, W//2], mode='bilinear'), 
        #        F.interpolate(h_relu4, [H//2, W//2], mode='bilinear')]
        # out = torch.cat(out, dim=1)
        return h_relu3


class CameraEncoder(nn.Module):
    def __init__(self, nc, nk, azi_scope, elev_range, dist_range):
        super(CameraEncoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=2)
        block2 = Conv2dBlock(32, 64, nk, stride=2, padding=2)
        block3 = Conv2dBlock(64, 128, nk, stride=2, padding=2)
        block4 = Conv2dBlock(128, 256, nk, stride=2, padding=2)
        block5 = Conv2dBlock(256, 128, nk, stride=2, padding=2)

        avgpool = nn.AdaptiveAvgPool2d(1)

        linear1 = self.linearblock(128, 64)
        linear2 = self.linearblock(64, 32)
        self.linear3 = nn.Linear(32, 4)

        #################################################
        all_blocks = [block1, block2, block3, block4, block5, avgpool]
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, \
        linear1, linear2 \

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU(inplace=True)
        ]
        return block2

    def atan2(self, y, x):
        r = torch.sqrt(x**2 + y**2 + 1e-12) + 1e-6
        phi = torch.sign(y) * torch.acos(x / r) * 180.0 / math.pi
        return phi

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        camera_output = self.linear3(x)

        # cameras
        distances = self.dist_min + torch.sigmoid(camera_output[:, 0]) * (self.dist_max - self.dist_min)
        elevations = self.elev_min + torch.sigmoid(camera_output[:, 1]) * (self.elev_max - self.elev_min)

        azimuths_x = camera_output[:, 2]
        azimuths_y = camera_output[:, 3]
        # azimuths = 90.0 - self.atan2(azimuths_y, azimuths_x)
        azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

        cameras = [azimuths, elevations, distances]
        return cameras


class ShapeEncoder(nn.Module):
    def __init__(self, nc, nk, num_vertices):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=2)
        #block2 = Conv2dBlock(32, 64, nk, stride=2, padding=2)        
        #block3 = Conv2dBlock(64, 128, nk, stride=2, padding=2)
        #block4 = Conv2dBlock(128, 256, nk, stride=2, padding=2)
        #block5 = Conv2dBlock(256, 512, nk, stride=2, padding=2)
        block2 = [ResBlock_half(32), ResBlock(64)]
        block3 = [ResBlock_half(64), ResBlock(128)]
        block4 = [ResBlock_half(128), ResBlock(256)]
        block5 = [ResBlock_half(256), ResBlock(512)]

        avgpool = nn.AdaptiveAvgPool2d(1)

        linear1 = self.linearblock(512, 512)
        linear2 = self.linearblock(512, 1024)
        self.linear3 = nn.Linear(1024, self.num_vertices * 3)

        #################################################
        all_blocks = [block1, *block2, *block3, *block4, *block5, avgpool]
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU(inplace=True)
        ]
        return block2

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        x = self.linear3(x)

        delta_vertices = x.view(batch_size, self.num_vertices, 3)
        delta_vertices = torch.tanh(delta_vertices)
        return delta_vertices


class LightEncoder(nn.Module):
    def __init__(self, nc, nk):
        super(LightEncoder, self).__init__()

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=2)
        block2 = Conv2dBlock(32, 64, nk, stride=2, padding=2)
        block3 = Conv2dBlock(64, 128, nk, stride=2, padding=2)
        block4 = Conv2dBlock(128, 256, nk, stride=2, padding=2)
        block5 = Conv2dBlock(256, 128, nk, stride=2, padding=2)

        avgpool = nn.AdaptiveAvgPool2d(1)

        linear1 = self.linearblock(128, 32)
        linear2 = self.linearblock(32, 32)
        self.linear3 = nn.Linear(32, 9)

        #################################################
        all_blocks = [block1, block2, block3, block4, block5, avgpool]
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU(inplace=True)
        ]
        return block2

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)

        lightparam = torch.tanh(x)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias
    
        return lightparam

class TextureEncoder(nn.Module):
    def __init__(self, nc, nf, nk, num_vertices):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = Conv2dBlock(nc, nf//2, nk, 2, 2, norm='bn')
        block2 = Conv2dBlock(nf//2, nf, nk, 2, 2, norm='bn')
        block3 = Conv2dBlock(nf, nf * 2, nk, 2, 2, norm='bn')
        block4 = Conv2dBlock(nf * 2, nf * 4, nk, 2, 2, norm='bn')
        block5 = Conv2dBlock(nf * 4, nf * 8, nk, 2, 2, norm='bn')

        avgpool = nn.AdaptiveAvgPool2d(1)

        linear1 = Conv2dBlock(nf * 8, nf * 16, 1, 1, 0, norm='bn')
        linear2 = Conv2dBlock(nf * 16, nf * 8, 1, 1, 0, norm='bn', activation = 'none')

        #################################################
        all_blocks = [block1, block2, block3, block4, block5, avgpool]
        self.encoder1 = nn.Sequential(*all_blocks)
        #model_ft = Resnet50_4C()
        #self.encoder1 = nn.Sequential(*[model_ft, avgpool])

        all_blocks = [linear1, linear2]
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        self.texture_flow = nn.Sequential(
            # input is Z, going into a convolution
            nn.Upsample(scale_factor=4),
            Conv2dBlock(nf * 8, nf * 4, 3, 1, 1, norm='bn', padding_mode='reflect'),
            #ResBlock(nf * 8, norm='bn', padding_mode='reflect'),
            # state size. (nf*8) x 8 x 8
            nn.Upsample(scale_factor=2),
            Conv2dBlock(nf * 4, nf * 2, 3, 1, 1, norm='bn', padding_mode='reflect'),
            #ResBlock(nf * 8, norm='bn', padding_mode='reflect'),
            # state size. (nf*4) x 16 x 16
            nn.Upsample(scale_factor=2),
            Conv2dBlock(nf * 2, nf, 3, 1, 1, norm='bn', padding_mode='reflect'),
            # state size. (nf*2) x 32 x 32
            nn.Upsample(scale_factor=2),
            Conv2dBlock(nf, nf, 3, 1, 1, norm='bn', padding_mode='reflect'),
            #ResBlock(nf, norm='bn', padding_mode='reflect'),
            # state size. (nf) x 64 x 64
            nn.Upsample(scale_factor=2),
            Conv2dBlock(nf, nf, 3, 1, 1, norm='bn', padding_mode='reflect'),
            #ResBlock(nf, norm='bn', padding_mode='reflect'),
            # state size. (nf) x 128 x 128
            nn.Upsample(scale_factor=2),
            Conv2dBlock(nf, nf//2, 3, 1, 1, norm='bn', padding_mode='reflect'),
            # state size. (nf) x 256 x 256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf//2, 2, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        img = x[:, :3]
        batch_size = x.shape[0]
        x = self.encoder1(x)
        x = x.view(batch_size, -1, 1, 1)
        x = self.encoder2(x) # 32x256x1x1
        uv_sampler = self.texture_flow(x).permute(0, 2, 3, 1) # 32 x4x4x2
        textures = F.grid_sample(img, uv_sampler) # 32 x 3 x4x4

        textures_flip = textures.flip([2])
        textures = torch.cat([textures, textures_flip], dim=2)
        return textures

class Resnet50_4C(nn.Module):
    def __init__(self):
        super(Resnet50_4C, self).__init__()
        model = models.resnet50(pretrained=True)
        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        model.conv1.weight.data[:, :3] = weight
        model.conv1.weight.data[:, 3] = model.conv1.weight[:, 0]
        self.model = model
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x 

class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='relu', padding_mode='zeros', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if res_type=='basic':
            model += [Conv2dBlock(dim ,dim//2, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim//2 ,dim, 3, 1, 1, norm=norm, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', padding_mode=padding_mode)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlock_half(nn.Module):
    def __init__(self, dim, norm='bn', activation='relu', padding_mode='zeros', res_type='basic'):
        super(ResBlock_half, self).__init__()

        model = []
        if res_type=='basic':
            model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', padding_mode=padding_mode)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.model(x)
        out = torch.cat([out,residual], dim=1)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', padding_mode='zeros', dilation=1, fp16 = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding, padding_mode=padding_mode, dilation=dilation, bias=self.use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# class FeatEncoder(nn.Module):
#     """
#     output 3 levels of features using a FPN structure
#     """
#     def __init__(self, nc, nf):
#         super(FeatEncoder, self).__init__()

#         self.conv0 = nn.Sequential(
#                         ConvBnReLU(nc, nf, 3, 1, 1),
#                         ConvBnReLU(nf, nf, 3, 1, 1))

#         self.conv1 = nn.Sequential(
#                         ConvBnReLU(nf, 2*nf, 5, 2, 2),
#                         ConvBnReLU(2*nf, 2*nf, 3, 1, 1))

#         self.conv2 = nn.Sequential( 
#                         ConvBnReLU(2*nf, 4*nf, 5, 2, 2),
#                         ConvBnReLU(4*nf, 4*nf, 3, 1, 1))

#         self.toplayer = nn.Conv2d(4*nf, 4*nf, 1)
#         self.lat1 = nn.Conv2d(2*nf, 4*nf, 1)
#         self.lat0 = nn.Conv2d(nf, 4*nf, 1)

#         # to reduce channel size of the outputs from FPN
#         self.smooth1 = nn.Conv2d(4*nf, 2*nf, 3, padding=1)
#         self.smooth0 = nn.Conv2d(4*nf, nf, 3, padding=1)
        

#     def _upsample_add(self, x, y):
#         return F.interpolate(x, scale_factor=2, 
#                              mode="bilinear", align_corners=True) + y

#     def forward(self, x):
#         # x: (B, 3, H, W)
#         conv0 = self.conv0(x) # (B, 8, H, W)
#         conv1 = self.conv1(conv0) # (B, 16, H//2, W//2)
#         conv2 = self.conv2(conv1) # (B, 32, H//4, W//4)
#         feat2 = self.toplayer(conv2) # (B, 32, H//4, W//4)
#         feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//2, W//2)
#         feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H, W)

#         # reduce output channels
#         feat1 = self.smooth1(feat1) # (B, 16, H//2, W//2)
#         feat0 = self.smooth0(feat0) # (B, 8, H, W)
#         return feat0
