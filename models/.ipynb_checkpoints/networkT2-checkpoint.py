import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = pad_diff(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def pad_diff(x1, x2):
    # input is CHW
    diff_y = x2.size()[2] - x1.size()[2]
    diff_x = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    return x1


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False):
        super(UpConv, self).__init__()
        modules = []

        if deconv:
            modules.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1))
        else:
            modules += [nn.Upsample(scale_factor=2), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        modules += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*modules)

    def forward(self, x, pad_like=None):
        x1 = self.up(x)
        return pad_diff(x1, pad_like)


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        resampler = self.psi(psi)
        return x * resampler


class SingleConv(nn.Module):
    def __init__(self, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class RecurrentConv(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RecurrentConv, self).__init__()
        self.t = t
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.single_conv = SingleConv(out_channels)

    def forward(self, x):
        x_in = self.inconv(x)
        x_out = self.single_conv(x_in)
        for i in range(self.t):
            x_out = self.single_conv(x_in + x_out)
        return x_out


class R2UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(R2UNetBlock, self).__init__()
        self.t = t
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.single_conv1 = SingleConv(out_channels)
        self.single_conv2 = SingleConv(out_channels)

    def forward(self, x):
        x_in = self.inconv(x)

        # First Recurrent Block
        x_r1 = self.single_conv1(x_in)
        for i in range(self.t):
            x_r1 = self.single_conv1(x_in + x_r1)

        # Second Recurrent Block
        x_out = self.single_conv2(x_r1)
        for i in range(self.t):
            x_out = self.single_conv2(x_r1 + x_out)

        return x_in + x_out  # Residual + Recurrent


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        smooth = 1e-5
        output = torch.sigmoid(output)
        num = target.size(0)
        output = output.view(num, -1)
        target = target.view(num, -1)
        intersection = (output * target)
        dice = (2. * intersection.sum(1) + smooth) / (output.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class R2UNet(nn.Module):
    def __init__(self, in_channels, n_classes, t=2):
        super(R2UNet, self).__init__()

        n_filter = [64, 128, 256, 512, 1024]  # [32, 64, 128, 256, 512]  #

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.r2down1 = R2UNetBlock(in_channels, n_filter[0], t)
        self.r2down2 = R2UNetBlock(n_filter[0], n_filter[1], t)
        self.r2down3 = R2UNetBlock(n_filter[1], n_filter[2], t)
        self.r2down4 = R2UNetBlock(n_filter[2], n_filter[3], t)
        self.r2down5 = R2UNetBlock(n_filter[3], n_filter[4], t)

        self.up5 = UpConv(n_filter[4], n_filter[3], deconv=True)
        self.r2up5 = R2UNetBlock(n_filter[4], n_filter[3], t)
        self.dropout5 = nn.Dropout(p=0.4)
        
        self.up4 = UpConv(n_filter[3], n_filter[2], deconv=True)
        self.r2up4 = R2UNetBlock(n_filter[3], n_filter[2], t)
        self.dropout4 = nn.Dropout(p=0.3)

        self.up3 = UpConv(n_filter[2], n_filter[1], deconv=True)
        self.r2up3 = R2UNetBlock(n_filter[2], n_filter[1], t)
        self.dropout3 = nn.Dropout(p=0.2)

        self.up2 = UpConv(n_filter[1], n_filter[0], deconv=True)
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
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
