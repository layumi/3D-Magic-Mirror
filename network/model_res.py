import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from .utils import weights_init, weights_init_classifier
import math
import timm

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

######################################################################
class MMPool(nn.Module):
    # MMPool zhedong zheng
    def __init__(self, dim = 1, p=0., eps=1e-6):
        super(MMPool,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.mmpool(x, p=self.p, eps=self.eps)

    def mmpool(self, x, p, eps):
        s = x.shape
        x_max = torch.nn.functional.adaptive_max_pool2d(x, output_size=(1,1))
        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
        w = torch.sigmoid(self.p)
        x = x_max*w + x_avg*(1-w)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'


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

class BackgroundEncoder(nn.Module):
    def __init__(self, nc, droprate=0.0): # input.shape == output.shape rgb 3 channel
        super(BackgroundEncoder, self).__init__()
        all_blocks = [Conv2dBlock(3, 32, 3, 2, 1, norm='none', activation='none', padding_mode='zeros'),
                  ResBlock(32, norm='none'), 
                  ResBlock(32, norm='none'), 
                  ResBlock(32, norm='none'), 
                  nn.Upsample(scale_factor=2),
                  nn.Dropout2d(droprate/2), #small drop for dense prediction
                  Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                  nn.Sigmoid()]
        self.encoder = nn.Sequential(*all_blocks)
        self.encoder.apply(weights_init)
        self.encoder[-2].apply(weights_init_classifier)

    def forward(self, x):
        img = x[:, :3]
        mask = x[:, 3].unsqueeze(1)
        bg = img * (1-mask) #always 3 channel
        return self.encoder(bg)

class CameraEncoder(nn.Module):
    def __init__(self, nc, nk, azi_scope, elev_range, dist_range, droprate = 0.0):
        super(CameraEncoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2)
        block2 = Conv2dBlock(32, 64, nk, stride=2, padding=nk//2)
        block3 = Conv2dBlock(64, 128, nk, stride=2, padding=nk//2)
        block4 = Conv2dBlock(128, 256, nk, stride=2, padding=nk//2)
        block5 = Conv2dBlock(256, 128, nk, stride=2, padding=nk//2)

        #avgpool = nn.AdaptiveAvgPool2d(1)
        avgpool = MMPool()

        linear1 = self.linearblock(128, 32, relu = False)
        #linear2 = self.linearblock(32, 32, relu=False)
        self.linear3 = nn.Linear(32, 4)

        #################################################
        all_blocks = [block1, block2, block3, block4, block5, avgpool]

        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 #+ linear2
        if droprate>0:
            all_blocks += [nn.Dropout(p=droprate)]
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        self.encoder1.apply(weights_init)
        self.encoder2.apply(weights_init)
        self.linear3.apply(weights_init_classifier)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1 

    def linearblock(self, indim, outdim, relu=True):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
        ]
        if relu:
            block2.append(nn.ReLU(inplace=True))
        return block2

    def atan2(self, y, x):
        r = torch.sqrt(x**2 + y**2 + 1e-12) + 1e-6
        phi = torch.sign(y) * torch.acos(x / r) * 180.0 / math.pi
        return phi

    def forward(self, x):
        bnum = x.shape[0]
        x = self.encoder1(x)
        x = x.view(bnum, -1)
        x = self.encoder2(x)

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
    def __init__(self, nc, nk, num_vertices, pretrain='none', droprate=0.0):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices

        if pretrain=='none':
            block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2)
            block2 = [ResBlock_half(32), ResBlock(64)]
            block3 = [ResBlock_half(64), ResBlock(128), ResBlock(128)]
            block4 = [ResBlock_half(128), ResBlock(256), ResBlock(256)]
            block5 = [ResBlock_half(256), ResBlock(512)]

            avgpool = MMPool()
            all_blocks = [block1, *block2, *block3, *block4, *block5, avgpool]
            self.encoder1 = nn.Sequential(*all_blocks)
            self.encoder1.apply(weights_init)
            in_dim = 512
        elif pretrain=='res18':
            avgpool = MMPool()
            self.encoder1 = nn.Sequential(*[Resnet_4C(pretrain), avgpool])
            in_dim = 512
        elif pretrain=='res50':
            avgpool = MMPool()
            self.encoder1 = nn.Sequential(*[Resnet_4C(pretrain), avgpool])
            in_dim = 2048
        elif pretrain=='hr18':
            avgpool = MMPool()
            self.encoder1 = nn.Sequential(*[HRnet_4C(), avgpool])
            in_dim = 2048
        else: 
            Print('unknown network')
        #################################################
        linear1 = self.linearblock(in_dim, 1024, relu = False)
        #linear2 = self.linearblock(512, 1024, relu = False)

        all_blocks = linear1 
        if droprate>0:
            all_blocks += [nn.Dropout(p=droprate)]
        self.encoder2 = nn.Sequential(*all_blocks)
        self.encoder2.apply(weights_init)

        #################################################
        self.linear3 = nn.Linear(1024, self.num_vertices * 3)
        self.linear3.apply(weights_init_classifier)

    def linearblock(self, indim, outdim, relu=True):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
        ]
        if relu:
            block2.append(nn.ReLU(inplace=True))
        return block2

    def forward(self, x):
        bnum = x.shape[0]
        x = self.encoder1(x)
        x = x.view(bnum, -1)
        x = self.encoder2(x)
        x = self.linear3(x)

        delta_vertices = x.view(bnum, self.num_vertices, 3)
        delta_vertices = torch.tanh(delta_vertices)
        return delta_vertices


class LightEncoder(nn.Module):
    def __init__(self, nc, nk, droprate = 0.0):
        super(LightEncoder, self).__init__()

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2)
        block2 = Conv2dBlock(32, 64, nk, stride=2, padding=nk//2)
        block3 = Conv2dBlock(64, 128, nk, stride=2, padding=nk//2)
        block4 = Conv2dBlock(128, 256, nk, stride=2, padding=nk//2)
        block5 = Conv2dBlock(256, 128, nk, stride=2, padding=nk//2)

        #avgpool = nn.AdaptiveAvgPool2d(1)
        avgpool = MMPool()

        linear1 = self.linearblock(128, 32, relu=False)
        #linear2 = self.linearblock(32, 32)
        self.linear3 = nn.Linear(32, 9)

        #################################################
        all_blocks = [block1, block2, block3, block4, block5, avgpool]
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 #+ linear2
        if droprate>0:
            all_blocks += [nn.Dropout(p=droprate)]
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        self.encoder1.apply(weights_init)
        self.encoder2.apply(weights_init)
        self.linear3.apply(weights_init_classifier)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1

    def linearblock(self, indim, outdim, relu=True):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
        ]
        if relu:
            block2.append(nn.ReLU(inplace=True))
        return block2

    def forward(self, x):
        bnum = x.shape[0]
        x = self.encoder1(x)
        x = x.view(bnum, -1)
        x = self.encoder2(x)
        x = self.linear3(x)

        lightparam = torch.tanh(x)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias
    
        return lightparam

class TextureEncoder(nn.Module):
    def __init__(self, nc, nf, nk, num_vertices, ratio=1, makeup=0, droprate = 0 ):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.makeup = makeup
        self.block1 = Conv2dBlock(nc, 32, nk, 2, 2, norm='bn') # 256 -> 128*128*32
        self.block2 = nn.Sequential(*[ResBlock_half(32), ResBlock(64)]) # 128 -> 64*64*64
        self.block3 = nn.Sequential(*[ResBlock_half(64), ResBlock(128)]) # 64->32*32*128
        self.block4 = nn.Sequential(*[ResBlock_half(128), ResBlock(256)]) # 32 -> 16*16*256
        self.block5 = nn.Sequential(*[ResBlock_half(256), ResBlock(512)]) # 16-> 8*8*512
        #avgpool = MMPool()

        #################################################
        #model_ft = Resnet50_4C()
        #self.encoder1 = nn.Sequential(*[model_ft, avgpool])

        # 8*8*512
        up1 = [Conv2dBlock(512, 256, 3, 1, 1, norm='bn', padding_mode='zeros'), ResBlock(256), nn.Upsample(scale_factor=2)]
        # 16*16*256 + 16*16*256 = 16*16*512
        up2 = [Conv2dBlock(512, 128, 3, 1, 1, norm='bn', padding_mode='zeros'), ResBlock(128), nn.Upsample(scale_factor=2)]
        # 32*32*128 + 32*32*128 =  32*32*256
        up3 = [Conv2dBlock(256, 64, 3, 1, 1, norm='bn', padding_mode='zeros'), ResBlock(64), nn.Upsample(scale_factor=2)]
        # 64*64*64 + 64*64*64 = 64*64*128 
        up4 = [Conv2dBlock(128, 64, 3, 1, 1, norm='bn', padding_mode='zeros'), ResBlock(64), nn.Upsample(scale_factor=2)]
        # 128*128*64
        up5 = [Conv2dBlock(64, 32, 3, 1, 1, norm='bn', padding_mode='zeros'), ResBlock(32), nn.Upsample(scale_factor=2)]
        # 256*256
        up6 = [Conv2dBlock(32, 2, 3, 1, 1, norm='none',  activation='none', padding_mode='zeros'), nn.Tanh()]
        if droprate >0:
            up6 = [nn.Dropout2d(droprate/2)] + up6 # small drop for dense prediction

        if self.makeup==1:
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 3, 1, 1, norm='in', padding_mode='zeros'),
                                      ResBlock(32, norm='in'), ResBlock(32, norm='in'),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      nn.Sigmoid()])
        elif self.makeup==2:
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 3, 1, 1, norm='bn', padding_mode='zeros'),
                                      ResBlock(32, norm='bn'), ResBlock(32, norm='bn'),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      nn.Sigmoid()])
        elif self.makeup==3:
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 3, 1, 1, norm='ln', padding_mode='zeros'),
                                      ResBlock(32, norm='ln'), ResBlock(32, norm='ln'),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      nn.Sigmoid()])
        elif self.makeup==4:
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 3, 1, 1, norm='none', padding_mode='zeros'),
                                      ResBlock(32, norm='none'), ResBlock(32, norm='none'),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      nn.Sigmoid()])


        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)
        self.up3 = nn.Sequential(*up3)
        self.up4 = nn.Sequential(*up4)
        self.up5 = nn.Sequential(*up5)
        self.up6 = nn.Sequential(*up6)

        # Initialize with Xavier Glorot
        self.block1.apply(weights_init)
        self.block2.apply(weights_init)
        self.block3.apply(weights_init)
        self.block4.apply(weights_init)
        self.block5.apply(weights_init)
        self.up1.apply(weights_init)
        self.up2.apply(weights_init)
        self.up3.apply(weights_init)
        self.up4.apply(weights_init)
        self.up5.apply(weights_init)
        self.up6.apply(weights_init_classifier)
        if self.makeup:
            self.make.apply(weights_init)
            self.make[-2].apply(weights_init_classifier)

    def forward(self, x):
        img = x[:, :3]
        batch_size = x.shape[0]
        # down
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        # up 
        up1 = self.up1(x5)
        up2 = self.up2(torch.cat((up1,x4),dim=1))
        up3 = self.up3(torch.cat((up2,x3),dim=1))
        up4 = self.up4(torch.cat((up3,x2),dim=1))
        up5 = self.up5(up4)
        texture_flow = self.up6(up5)
        # clear
        del x1,x2,x3,x4,x5, up1,up2,up3,up4,up5
        uv_sampler = texture_flow.permute(0, 2, 3, 1) # 32 x256x256x2
        textures = F.grid_sample(img, uv_sampler, align_corners=False) # 32 x 3 x128x128

        # zzd: Here we need a network to make up the hole via reasonable guessing.
        if self.makeup:
            textures = self.make(textures)

        textures_flip = textures.flip([2])
        textures = torch.cat([textures, textures_flip], dim=2)
        return textures

class Resnet_4C(nn.Module):
    def __init__(self, pretrain):
        super(Resnet_4C, self).__init__()
        if pretrain == 'res50':
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)
        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        model.conv1.weight.data[:, :3] = weight
        model.conv1.weight.data[:, 3] = model.conv1.weight[:, 0]

        model.layer4[0].downsample[0].stride = (1,1)
        model.layer4[0].conv1.stride = (1,1)
        model.layer4[0].conv2.stride = (1,1)
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

class HRnet_4C(nn.Module):
    def __init__(self):
        super(HRnet_4C, self).__init__()
        model = timm.create_model('hrnet_w18', pretrained=True)
        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False) #here 4 indicates 4-channel input
        model.conv1.weight.data[:, :3] = weight
        model.conv1.weight.data[:, 3] = model.conv1.weight[:, 0]
        self.model = model
    def forward(self, x):
        x = self.model.forward_features(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
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
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
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
                 padding=0, norm='none', activation='lrelu', padding_mode='zeros', dilation=1, fp16 = False):
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

