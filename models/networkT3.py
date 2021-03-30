import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class ResNeSt(nn.Module):
    def __init__(self, arch='resnest101', pretrained=True):
        super().__init__()
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # load pretrained ResNeSt-101 models
        resneSt = torch.hub.load('zhanghang1989/ResNeSt', arch, pretrained=pretrained)
        
        self.layer0 = nn.Sequential(*list(resneSt.children()))[:4] # extracting pre-trained layers
        self.bottle_layer1 = nn.Sequential(*list(resneSt.children()))[4]
        self.bottle_layer2 = nn.Sequential(*list(resneSt.children()))[5]
        self.bottle_layer3 = nn.Sequential(*list(resneSt.children()))[6]
        self.bottle_layer4 = nn.Sequential(*list(resneSt.children()))[7]
        
        
    def forward(self, input):
        x = self.layer0(input)
        x = self.bottle_layer1(x)
        low_level_feat = x
        x = self.bottle_layer2(x)
        x = self.bottle_layer3(x)
        x = self.bottle_layer4(x)
        
        return x, low_level_feat


# ##############

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        # add doc
        # checkout atrous conv
        super().__init__()
        # using dilation to realize atrous convolution -> introduce space between kernel points
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, inplanes=2048, workplanes=256):
        """
        Args:
            inplanes: number of channels for encoder (backbone)
        """
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18] # stride=16 
            
        self.aspp1 = ASPPModule(inplanes, workplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, workplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, workplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, workplanes, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, workplanes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(workplanes),
                                             nn.ReLU())
        
        self.conv1 = nn.Conv2d(workplanes*5, workplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(workplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# ###########

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes = 256, workplanes=256):
        """
        Args:
            low_level_inplanes: number of channels in output from encoder/backbone
        """
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        
        self.upsample = nn.ConvTranspose2d(workplanes, workplanes, kernel_size=2, stride=2)
        # 256 + 48
        self.last_conv = nn.Sequential(nn.Conv2d(workplanes+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.3), # increase
                                       nn.Conv2d(128, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        # x from ASPP (ch=256)
        # low_level_fea from encoder/backbone (ch=256)
        low_level_feat = self.conv1(low_level_feat) # 256->48
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        # match dim
        x = self.upsample(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLab(nn.Module):
    def __init__(self, backbone='resnest', output_stride=16, num_classes=21, freeze_bn=False):
        super(DeepLab, self).__init__()
        
        self.backbone = ResNeSt() # encoder            
        self.aspp = ASPP(output_stride, workplanes=256, inplanes=2048)
        self.decoder = Decoder(num_classes, workplanes=256)

    def forward(self, input):
        x, low_level_feat = self.backbone(input) # low_level_feat -> cross connection from encoder to decoder
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True) # match output size

        return x

if __name__ == "__main__":
    from torchsummary import summary
    model = DeepLab(output_stride=16, num_classes=19)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())
    summary(model, input, device=torch.device('cuda:6'))


