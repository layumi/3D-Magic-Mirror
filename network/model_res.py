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
    def __init__(self, shape=(1,1), dim = 1, p=0., eps=1e-6):
        super(MMPool,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
        self.shape = shape

    def forward(self, x):
        return self.mmpool(x, shape = self.shape,  p=self.p, eps=self.eps)

    def mmpool(self, x, shape, p, eps):
        s = x.shape
        x_max = torch.nn.functional.adaptive_max_pool2d(x, output_size=shape)
        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, output_size=shape)
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
    def __init__(self, nc, droprate=0.0, coordconv=False): # input.shape == output.shape rgb 3 channel
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
    def __init__(self, nc, nk, azi_scope, elev_range, dist_range, droprate = 0.0, coordconv=False, norm = 'bn', ratio=1, pretrain='none'):
        super(CameraEncoder, self).__init__()

        self.azi_scope = float(azi_scope)
        self.ratio = ratio
        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])

        #if pretrain=='none':
            # 2-4-4-3 = 12 resblocks = 24 conv
        self.encoder1 = Base_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
        self.encoder1.apply(weights_init)
        in_dim = 288
        #elif pretrain=='unet': #unet from scratch
        #    self.encoder1 = UNet_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
        #    in_dim = 32
        #elif pretrain=='res18':
        #    self.encoder1 = Resnet_4C(pretrain)
        #    in_dim = 512
        #elif pretrain=='res50':
        #    self.encoder1 = Resnet_4C(pretrain)
        #    in_dim = 2048
        #elif 'hr18' in pretrain:
        #    self.encoder1 = HRnet_4C(pretrain = 'hr18sv1') # set as small hrnet for fast inference.
        #    in_dim = 2048
        #else:
        #    print('unknown network')

        #avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = MMPool()

        linear1 = self.linearblock(in_dim, 128, relu = False)
        #linear2 = self.linearblock(32, 32, relu=False)
        self.linear3 = nn.Linear(128, 6)

        all_blocks = linear1 #+ linear2
        if droprate>0:
            all_blocks += [nn.Dropout(p=droprate)]
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        self.encoder2.apply(weights_init)
        self.linear3.apply(weights_init_classifier)

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
        # -180 ~ 180
        return phi

    def forward(self, x):
        bnum = x.shape[0]
        x = self.encoder1(x)
        x = self.avgpool(x)
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

        biases_x = torch.tanh(camera_output[:, 4]).unsqueeze(-1) # x from -1 to 1 #
        biases_y = torch.tanh(camera_output[:, 5]).unsqueeze(-1) # * self.ratio # y from -2 to 2
        biases = torch.cat( (biases_x, biases_y), dim=1)
        # y from -2 to 2
        cameras = [azimuths, elevations, distances, biases]
        return cameras


class ShapeEncoder(nn.Module):
    def __init__(self, nc, nk, num_vertices, pretrain='none', droprate=0.0, coordconv=False, norm = 'bn'):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.mmpool = MMPool((1,1))
        if pretrain=='none':
            # 2-4-4-3 = 12 resblocks = 24 conv
            self.encoder1 = Base_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
            self.encoder1.apply(weights_init)
            in_dim = 288 
        elif pretrain=='unet': #unet from scratch 
            self.encoder1 = UNet_4C(nc=nc, nk=nk, norm = norm, coordconv=coordconv)
            in_dim = 32
        elif pretrain=='res18':
            self.encoder1 = Resnet_4C(pretrain)
            in_dim = 512 
        elif pretrain=='res50':
            self.encoder1 = Resnet_4C(pretrain)
            in_dim = 2048 
        elif 'hr18' in pretrain:
            self.encoder1 = HRnet_4C(pretrain)
            in_dim = 2048 
        else: 
            print('unknown network')
        #################################################
        linear1 = self.Conv1d(in_dim*3 + 3, 256, relu=True, droprate = droprate, coordconv=True )
        linear2 = self.Conv1d(256, 3, relu = False)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)
        self.encoder2.apply(weights_init)
        #################################################
        self.linear3 = nn.Linear(self.num_vertices * 3, self.num_vertices * 3)
        self.linear3.apply(weights_init_classifier)

    def linearblock(self, indim, outdim, relu=True):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
        ]
        if relu:
            block2.append(nn.ReLU(inplace=True))
        return block2

    def Conv1d(self, indim, outdim, relu=True, droprate = 0.0, coordconv=False):
        if coordconv: 
            indim = indim + 1
        block2 = [
            nn.Conv1d(indim, outdim, kernel_size=1),
            nn.BatchNorm1d(outdim),
        ]
        if relu:
            block2.append(nn.LeakyReLU(0.2, inplace=True))
        if droprate>0:
            block2.append(nn.Dropout(p=droprate))
        if coordconv:
            block2 = [AddCoords1d()]+ block2
        return block2

    def forward(self, x, template, lpl):
        bnum = x.shape[0]
        x = self.encoder1(x) # recommend a high resolution  8x4
        # template is 1x642x3, use location (x,y) to get local feature
        #print(x.shape)
        current_position = template.repeat(bnum,1,1).view(bnum, self.num_vertices, 1 , 3) # 32x642x1x3
        uv_sampler = current_position[:,:,:,0:2].cuda().detach() # 32 x642x1x2
        # depth = current_position[:,:,:,2].cuda().detach() # 32 x642x1
        # extract local 
        local = F.grid_sample(x, uv_sampler, mode='bilinear', align_corners=False) # 32 x 288 x642x1
        glob = self.mmpool(x) # mean + max pool
        glob = glob.repeat(1,1,self.num_vertices,1)
        neighbor_diff = torch.mm(local.view(-1, self.num_vertices), lpl.cuda()) # 32x288x642 * 642x642
        neighbor_diff = neighbor_diff.view(bnum, -1, self.num_vertices, 1)
        # Per-Point: local + global + neighbor_diff + xyz
        x = torch.cat( (local, glob, neighbor_diff, current_position.permute(0,3,1,2)), dim = 1 ) # 32x (288*3+3) x642x1
        x = self.encoder2(x.squeeze()) # 32x3x642
        delta_vertices = x.permute(0, 2, 1).reshape(bnum, -1) # 32x (642x3)
        delta_vertices = self.linear3(delta_vertices) # all points. init is close to 0
        delta_vertices = 0.5 * torch.tanh(delta_vertices) # limit the bias within [-1, 1]
        #print(delta_vertices.shape)
        return delta_vertices.view(bnum, self.num_vertices, 3)


class LightEncoder(nn.Module):
    def __init__(self, nc, nk, droprate = 0.0, coordconv=False, norm='bn'):
        super(LightEncoder, self).__init__()

        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2, norm = norm, coordconv=coordconv)
        block2 = Conv2dBlock(32, 64, nk, stride=2, padding=nk//2, norm = norm, coordconv=coordconv)
        block3 = Conv2dBlock(64, 128, nk, stride=2, padding=nk//2, norm = norm)
        block4 = Conv2dBlock(128, 256, nk, stride=2, padding=nk//2, norm = norm)
        block5 = Conv2dBlock(256, 128, nk, stride=2, padding=nk//2, norm = norm)

        #avgpool = nn.AdaptiveAvgPool2d(1)
        avgpool = MMPool()

        linear1 = self.linearblock(128, 64, relu=False)
        #linear2 = self.linearblock(32, 32)
        self.linear3 = nn.Linear(64, 9)

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
    def __init__(self, nc, nf, nk, num_vertices, ratio=1, makeup=0, droprate = 0, coordconv=False, norm='bn' ):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.makeup = makeup
        self.block1 = Conv2dBlock(nc, 32, nk, 2, 2, norm='bn', coordconv=coordconv) # 256 -> 128*128*32
        # 2-4-4-2
        self.block2 = nn.Sequential(*[ResBlock_half(32, norm=norm), ResBlock(64, norm=norm)]) # 128 -> 64*64*64
        self.block3 = nn.Sequential(*[ResBlock_half(64, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm)]) # 64->32*32*128
        self.block4 = nn.Sequential(*[ResBlock_half(128, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm)]) # 32 -> 16*16*256
        self.block5 = nn.Sequential(*[ResBlock_half(256, norm=norm), ResBlock(512, norm=norm)]) # 16-> 8*8*512
        #avgpool = MMPool()

        #################################################
        #model_ft = Resnet50_4C()
        #self.encoder1 = nn.Sequential(*[model_ft, avgpool])

        # 8*8*512
        up1 = [Conv2dBlock(512, 256, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(256), nn.Upsample(scale_factor=2)]
        # 16*16*256 + 16*16*256 = 16*16*512
        up2 = [Conv2dBlock(512, 128, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(128), nn.Upsample(scale_factor=2)]
        # 32*32*128 + 32*32*128 =  32*32*256
        up3 = [Conv2dBlock(256, 64, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(64), nn.Upsample(scale_factor=2)]
        # 64*64*64 + 64*64*64 = 64*64*128 
        up4 = [Conv2dBlock(128, 64, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(64), nn.Upsample(scale_factor=2)]
        # 128*128*64
        up5 = [Conv2dBlock(64, 32, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(32), nn.Upsample(scale_factor=2)]
        # 256*256
        up6 = [Conv2dBlock(32, 2, 3, 1, 1, norm='none',  activation='none', padding_mode='zeros'), nn.Tanh()]
        if droprate >0:
            up6 = [nn.Dropout(droprate/2)] + up6 # small drop for dense prediction. Dropout 2D may be too strong. so I still use dropout

        if self.makeup==1: # identify
            self.make = nn.Sequential(*[nn.Dropout2d(droprate), # drop at the begin
                                      Conv2dBlock(3, 32, 5, 1, 2, norm='none', activation = 'lrelu', padding_mode='zeros', coordconv = coordconv),
                                      ResBlock(32, norm='in'), ResBlock(32, norm='in'),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      ])
        elif self.makeup==2: # dropout 1d
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 5, 1, 2, norm='none', activation = 'lrelu', padding_mode='zeros', coordconv = coordconv),
                                      ResBlock(32, norm='in'), ResBlock(32, norm='in'),
                                      nn.Dropout2d(droprate), # drop in the last
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      ])
        elif self.makeup==3: # ln
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 5, 1, 2, norm='ln', activation = 'lrelu', padding_mode='zeros', coordconv = coordconv),
                                      #ResBlock(32, norm='ln'), ResBlock(32, norm='ln'),
                                      nn.Dropout2d(droprate),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      ])
        elif self.makeup==4: # in 
            self.make = nn.Sequential(*[Conv2dBlock(3, 32, 5, 1, 2, norm='in', activation = 'lrelu', padding_mode='zeros', coordconv = coordconv),
                                      #ResBlock(32, norm='none'), ResBlock(32, norm='none'),
                                      nn.Dropout2d(droprate),
                                      Conv2dBlock(32, 3, 3, 1, 1, norm='none', activation='none', padding_mode='zeros'),
                                      ])
        elif self.makeup==5: # identity mapping
            self.make = nn.Sequential()
            # remove the last tahnh
            up6 = up6[:-1]

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
            #self.make[-2].apply(weights_init_classifier)

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
        y = self.up1(x5)
        y = self.up2(torch.cat((y,x4),dim=1))
        y = self.up3(torch.cat((y,x3),dim=1))
        y = self.up4(torch.cat((y,x2),dim=1))
        y = self.up5(y)
        texture_flow = self.up6(y)
        # clear
        del x1,x2,x3,x4,x5,y
        uv_sampler = texture_flow.permute(0, 2, 3, 1) # 32 x256x256x2
        textures = F.grid_sample(img, uv_sampler, align_corners=False) # 32 x 3 x128x128
        textures_flip = textures.flip([2])
        textures = torch.cat([textures, textures_flip], dim=2)
        # zzd: Here we need a network to make up the hole via re-fining.
        # back may be differ from front.
        if self.makeup:
            textures = textures + self.make(textures)
            textures = torch.clamp(textures, min=0.0, max=1.0)
        #print(torch.max(textures[:]), torch.min(textures[:]))
        return textures


class Base_4C(nn.Module):
    def __init__(self, nc=4, nk=5, norm = 'bn', coordconv=True):
        super(Base_4C, self).__init__()
        # 2-4-4-3 = 12 resblocks = 24 conv
        block1 = Conv2dBlock(nc, 36, nk, stride=2, padding=nk//2, coordconv=coordconv)  #128 -> 64
        block2 = [ResBlock_half(36, norm=norm), ResBlock(72, norm=norm)] #64 -> 32
        block3 = [ResBlock_half(72, norm=norm), ResBlock(144, norm=norm), ResBlock(144, norm=norm), ResBlock(144, norm=norm)]  #32 -> 16
        block4 = [ResBlock_half(144, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm)] #16 -> 8
        block5 = [ResBlock(288, norm=norm), ResBlock(288, norm=norm), ResBlock(288, norm=norm)] #8->8

        all_blocks = [block1, *block2, *block3] #, avgpool]
        self.layer3 = nn.Sequential(*all_blocks)
        self.layer4 = nn.Sequential(*block4)
        self.layer5 = nn.Sequential(*block5)
        # 8*8*512
        self.layer3.apply(weights_init)
        self.layer4.apply(weights_init)
        self.layer5.apply(weights_init)
    def forward(self, x):
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4) 
        return x4 + x5

class UNet_4C(nn.Module):
    def __init__(self, nc=4, nk=5, norm = 'bn', coordconv=True):
        super(UNet_4C, self).__init__()
        # 2-4-4-3 = 12 resblocks = 24 conv
        block1 = Conv2dBlock(nc, 32, nk, stride=2, padding=nk//2, coordconv=coordconv)  #128 -> 64
        block2 = [ResBlock_half(32, norm=norm), ResBlock(64, norm=norm)] #64 -> 32
        block3 = [ResBlock_half(64, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm), ResBlock(128, norm=norm)]  #32 -> 16
        block4 = [ResBlock_half(128, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm), ResBlock(256, norm=norm)] #16 -> 8
        block5 = [ResBlock_half(256, norm=norm), ResBlock(512, norm=norm), ResBlock(512, norm=norm)] #8->4

        all_blocks = [block1, *block2, ] #, avgpool]
        self.layer2 = nn.Sequential(*all_blocks)
        self.layer3 = nn.Sequential(*block3)
        self.layer4 = nn.Sequential(*block4)
        self.layer5 = nn.Sequential(*block5)
        # 4*2*512
        up1 = [Conv2dBlock(512, 256, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(256), nn.Upsample(scale_factor=2)]
        # 8*4*256 + 8*4*256 
        up2 = [Conv2dBlock(512, 128, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(128), nn.Upsample(scale_factor=2)]
        # 32*32*128 + 32*32*128 =  32*32*256
        up3 = [Conv2dBlock(256, 64, 3, 1, 1, norm=norm, padding_mode='zeros', coordconv=coordconv), ResBlock(64), nn.Upsample(scale_factor=2)]
        up4 = [Conv2dBlock(128, 32, 3, 1, 1, norm='none',  activation='none', padding_mode='zeros'), ResBlock(32)]

        self.layer2.apply(weights_init)
        self.layer3.apply(weights_init)
        self.layer4.apply(weights_init)
        self.layer5.apply(weights_init)
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)
        self.up3 = nn.Sequential(*up3)
        self.up4 = nn.Sequential(*up4)
        self.up1.apply(weights_init)
        self.up2.apply(weights_init)
        self.up3.apply(weights_init)
        self.up4.apply(weights_init)


    def forward(self, x):
        x2 = self.layer2(x) # 64 channel
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4) 
        y = self.up1(x5) # 256 channel
        y = self.up2(torch.cat((y,x4),dim=1)) # 128 channel
        y = self.up3(torch.cat((y,x3),dim=1)) # 64 channel
        y = self.up4(torch.cat((y,x2),dim=1)) # 32 channel

        return y


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
        model.conv1.weight.data[:, 3] = torch.mean(weight, dim=-1)

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
    def __init__(self, pretrain):
        super(HRnet_4C, self).__init__()
        if pretrain == 'hr18':
            model = timm.create_model('hrnet_w18', pretrained=True)
        elif pretrain == 'hr18sv2':
            model = timm.create_model('hrnet_w18_small_v2', pretrained=True)
        elif pretrain == 'hr18sv1':
            model = timm.create_model('hrnet_w18_small', pretrained=True)
        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False) #here 4 indicates 4-channel input
        model.conv1.weight.data[:, :3] = weight
        model.conv1.weight.data[:, 3] = torch.mean(weight, dim=-1) #model.conv1.weight[:, 0]
        self.model = model
    def forward(self, x):
        x = self.model.forward_features(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if norm == 'ibn':
           norm2 = 'bn'
        else: 
           norm2 = norm
        if res_type=='basic':
            model += [Conv2dBlock(dim ,dim//2, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim//2 ,dim, 3, 1, 1, norm=norm2, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm2, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', padding_mode=padding_mode)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += 0.2 * residual # to help initial learning
        return out

class ResBlock_half(nn.Module):
    def __init__(self, dim, norm='bn', activation='lrelu', padding_mode='zeros', res_type='basic'):
        super(ResBlock_half, self).__init__()

        model = []
        if norm == 'ibn':
           norm2 = 'bn'
        else:
           norm2 = norm
        if res_type=='basic':
            model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm2, activation='none', padding_mode=padding_mode)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, padding_mode=padding_mode)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm2, activation=activation, padding_mode=padding_mode)]
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

class AddCoords1d(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim)
        """
        batch_size, _, x_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).view(1, 1, x_dim)

        xx_channel = xx_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1)  # batchsize, 1, x_dim

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor)], dim=1)
        
        return ret
    
class AddCoords2d(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='lrelu', padding_mode='zeros', dilation=1, fp16 = False, coordconv = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # initialize convolution
        self.coordconv = coordconv
        if self.coordconv:
            input_dim = input_dim + 2
            self.addcoords = AddCoords2d(with_r=False)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding, padding_mode=padding_mode, dilation=dilation, bias=self.use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ibn':
            self.norm = IBN(norm_dim)
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
        if self.coordconv:
            x = self.addcoords(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True) # use affine.
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

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

