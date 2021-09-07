import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 3, 1, 3, bias=False),
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

