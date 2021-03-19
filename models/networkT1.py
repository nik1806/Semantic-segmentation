import torch
from torch import nn
from torchsummary import summary
import torchvision
from torchvision.models.resnet import resnet50

# pre-trained backbone
# resnet = torchvision.models.resnet.resnet50(pretrained=True)

device = torch.device('cuda:5')
summary(resnet.cuda(device), torch.rand((1, 3, 256, 256)).cuda(device))


class convBN(nn.Module):
    """
    A convolution block to help build layers.
    Structure: Conv -> batchnorm -> activation
    """

    def __init__(self, in_ch:int, out_ch:int, padding:int=1, ks:int=3, stride:int=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, padding=padding, kernel_size=ks, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU()
        self.selu = nn.SELU()
        # self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # if self.with_nonlinearity:
        x = self.selu(x)
        return x


# class Bridge(nn.Module):
#     """
#     This is the middle layer of the UNet which just consists of some
#     """

#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.bridge = nn.Sequential(
#             convBN(in_ch, out_ch),
#             convBN(out_ch, out_ch)
#         )

#     def forward(self, x):
#         return self.bridge(x)


class upBlock(nn.Module):
    """
    Perform an up-sampling step. Increase the height and width while reducing the channels (number)
    Structure: upsample -> convBN -> convBN
    """

    def __init__(self, in_ch:int, out_ch:int, up_conv_in_ch=None, up_conv_out_ch=None):
        """
        Args:
            in_ch: number of input channels to upsampling layer (output from previous upBlock)
            out_ch: number of output channels of upsampling layer
            up_conv_in_ch: number of input channels from down block
            up_conv_out_ch: number of output channels to next down block
        """

        super().__init__()

        if up_conv_in_ch == None:
            up_conv_in_ch = in_ch
        if up_conv_out_ch == None:
            up_conv_out_ch = out_ch

        # if upsampling_method == "conv_transpose":
        #     self.upsample = nn.ConvTranspose2d(up_conv_in_ch, up_conv_out_ch, kernel_size=2, stride=2)
        # elif upsampling_method == "bilinear":
        #     self.upsample = nn.Sequential(
        #         nn.Upsample(mode='bilinear', scale_factor=2),
        #         nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        #     )

        self.upsample = nn.ConvTranspose2d(up_conv_in_ch, up_conv_out_ch, kernel_size=2, stride=2)
        self.conv_blk_1 = convBN(in_ch, out_ch)
        self.conv_blk_2 = convBN(out_ch, out_ch)

    def forward(self, prev_x, down_x):
        """
        Return: upsampled feature map
        Args:
            prev_x: output from the previous up block
            down_x: output from the down block
        """
        x = self.upsample(prev_x)
        x = torch.cat([x, down_x], 1) # concatenate across channels
        x = self.conv_blk_1(x)
        x = self.conv_blk_2(x)
        return x


class UNet(nn.Module):
    """
        A UNet architecture with Resnet50 backbone (pre-trained)
    """

    def __init__(self, n_cls=21, depth=6):
        """
        Args:
            n_cls: total number of classes(output) for dataset/model
        """
        super().__init__()
        self.DEPTH = depth
        # pre-trained model provided by pytorch
        resnet = resnet50(pretrained=True)
        # list (save) layers for down/up blocks
        down_blocks = []
        up_blocks = []

        # DOWNBLOCKS -> increase channels
        self.input_block = nn.Sequential(*list(resnet.children()))[:3] # extracting layers
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()): # using bottleneck layers
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        # self.bridge = Bridge(2048, 2048)
        
        # Additional
        self.bridge = self.bridge = nn.Sequential(
            convBN(2048, 2048),
            convBN(2048, 2048)
        )

        # UPBLOCKS -> reduce channels
        up_blocks.append(upBlock(2048, 1024))
        up_blocks.append(upBlock(1024, 512))
        up_blocks.append(upBlock(512, 256))
        up_blocks.append(upBlock(in_ch=192, out_ch=128, up_conv_in_ch=256, up_conv_out_ch=128)) #128 + 64
        up_blocks.append(upBlock(in_ch=67, out_ch=64, up_conv_in_ch=128, up_conv_out_ch=64)) # 64 + 3
        self.up_blocks = nn.ModuleList(up_blocks)

        # managing variable output channels (classes)
        self.out = nn.Conv2d(64, n_cls, kernel_size=1, stride=1)

    def forward(self, x):
        pre_pools = dict() # record of layers to provide skip connections
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (self.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        # output_feature_map = x
        x = self.out(x)
<<<<<<< HEAD
        del pre_pools # clear garbage

        # if with_output_feature_map:
        #     return x, output_feature_map
        # else:
        return x

print("INIT")
model = UNet(n_cls=21).cuda()
inp = torch.rand((1, 3, 128, 128)).cuda()
print("To gpu complete")
# out = model(inp)
# visualization of model
summary(model, inp)
# print(out.shape)
=======
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

device = torch.device('cuda:5')

model = UNetWithResnet50Encoder().cuda(device)
inp = torch.rand((1, 3, 256, 256)).cuda(device)
# out = model(inp)

# visualization of model
summary(model, inp, verbose=1)
# print(out.shape)


>>>>>>> e5e2f922ee9e13d6cecc255e521e9fbfb9b0e798
