import torch
import torch.nn as nn
import torch.nn.functional as F

'Major blocks used in implementation of R2U-Net are Upconvolution block, basic single convolution-batchnorm-relu block and R2U-Net block(residual+recurrent)'

class Upconvblock(nn.Module):

    'Convolutional upsampling block'
    'in_channel - number of channels in input'
    'out_channel - number of channels in output'
    'deconv - if set True then we will use learnable convolution layers for sampling else we will perform interpolation'

    def __init__(self, in_channels, out_channels, deconv=False):
        super(Upconvblock, self).__init__()
        modules = []

        if deconv:
            modules.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1))
        else: # interpolation
            modules += [nn.Upsample(scale_factor=2), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        modules += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*modules)

    def forward(self, x, pad_like=None):
        x1 = self.up(x)

        diff_y = pad_like.size()[2] - x1.size()[2]
        diff_x = pad_like.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])
        return x1


class ConvBN(nn.Module):

    'Basic single convolution layer with batch norm and ReLU '
    'out_channels - number of channels in output'

    def __init__(self, out_channels):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class R2UNetBlock(nn.Module):
    'Residual+Recurrent block using convolution layers, conv-batchnorm-relu block'
    'in_channel - number of channels in input'
    'out_channel - number of channels in output'
    't - number of single recurrent blocks (2 here)'

    def __init__(self, in_channels, out_channels, t=2):
        super(R2UNetBlock, self).__init__()
        self.t = t
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.single_conv1 = ConvBN(out_channels)
        self.single_conv2 = ConvBN(out_channels)

    def forward(self, x):
        x_in = self.inconv(x)

        'Recurrent block 1'
        x_r1 = self.single_conv1(x_in)
        for i in range(self.t):
            x_r1 = self.single_conv1(x_in + x_r1)

        'Recurrent block 2'
        x_out = self.single_conv2(x_r1)
        for i in range(self.t):
            x_out = self.single_conv2(x_r1 + x_out)

        return x_in + x_out  # Residual + Recurrent


class R2UNet(nn.Module):
    'Main class for R2U-Net'
    'in_channel - number of channels in input'
    'n_classes = numeber of classes used in segmentation task. Here we are using 19 classes'
    't - number of single recurrent blocks (2 here)'

    def __init__(self, in_channels, n_classes, t=2):
        super(R2UNet, self).__init__()

        n_filter = [64, 128, 256, 512, 1024]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.r2down1 = R2UNetBlock(in_channels, n_filter[0], t)
        self.r2down2 = R2UNetBlock(n_filter[0], n_filter[1], t)
        self.r2down3 = R2UNetBlock(n_filter[1], n_filter[2], t)
        self.r2down4 = R2UNetBlock(n_filter[2], n_filter[3], t)
        self.r2down5 = R2UNetBlock(n_filter[3], n_filter[4], t)

        self.up5 = Upconvblock(n_filter[4], n_filter[3], deconv=True)
        self.r2up5 = R2UNetBlock(n_filter[4], n_filter[3], t)
        self.dropout5 = nn.Dropout(p=0.4)
        
        self.up4 = Upconvblock(n_filter[3], n_filter[2], deconv=True)
        self.r2up4 = R2UNetBlock(n_filter[3], n_filter[2], t)
        self.dropout4 = nn.Dropout(p=0.3)

        self.up3 = Upconvblock(n_filter[2], n_filter[1], deconv=True)
        self.r2up3 = R2UNetBlock(n_filter[2], n_filter[1], t)
        self.dropout3 = nn.Dropout(p=0.2)

        self.up2 = Upconvblock(n_filter[1], n_filter[0], deconv=True)
        self.r2up2 = R2UNetBlock(n_filter[1], n_filter[0], t)
        self.dropout2 = nn.Dropout(p=0.2)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        
        self._init_weight() # for better init and avoid overfitting

    def forward(self, x):
        x1 = self.r2down1(x)

        x2 = self.maxpool(x1)
        x2 = self.r2down2(x2)

        x3 = self.maxpool(x2)
        x3 = self.r2down3(x3)

        x4 = self.maxpool(x3)
        x4 = self.r2down4(x4)

        x5 = self.maxpool(x4)
        x5 = self.r2down5(x5)

        up5 = self.up5(x5, pad_like=x4)
        up5 = torch.cat((x4, up5), dim=1)
        up5 = self.r2up5(up5)
        up5 = self.dropout5(up5)
        
        up4 = self.up4(up5, pad_like=x3)
        up4 = torch.cat((x3, up4), dim=1)
        up4 = self.r2up4(up4)
        up4 = self.dropout4(up4)

        up3 = self.up3(up4, pad_like=x2)
        up3 = torch.cat((x2, up3), dim=1)
        up3 = self.r2up3(up3)
        up3 = self.dropout3(up3)

        up2 = self.up2(up3, pad_like=x1)
        up2 = torch.cat((x1, up2), dim=1)
        up2 = self.r2up2(up2)
        up2 = self.dropout2(up2)

        out = self.outconv(up2)
        return out
    
    'for weight initialization'

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight) #Fills the input Tensor with values
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
