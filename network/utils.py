import torch
import torch.nn as nn
from torch.nn import init

# custom weights initialization called on netE and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Block') == -1 and classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1: 
        init.normal_(m.weight.data, 1.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight')  and m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Block') == -1 and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_uniform_(m.weight.data)
        init.normal_(m.weight.data, std=0.00001) 
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

