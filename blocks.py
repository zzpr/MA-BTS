import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 定义下采样层
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_c=4) -> None:
        super(Encoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, 64,kernel_size=7,stride=1,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(64,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(128,[[2,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(256,[[2,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(512,[[2,1],[1,1]])
        
    def _make_layer(self, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out1 = self.conv1(x)     # [1, 4, 160, 160] -> [1, 64, 80, 80]
        out2 = self.conv2(out1)   # [1, 64, 80, 80] -> [1, 64, 80, 80]
        out3 = self.conv3(out2)   # [1, 64, 80, 80] -> [1, 128, 40, 40]
        out4 = self.conv4(out3)   # [1, 128, 40, 40] -> [1, 256, 20, 20]
        out5 = self.conv5(out4)   # [1, 256, 20, 20] -> [1, 512, 10, 10]
        return out3, out5


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rb = ResBlock(in_c, out_c)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.rb(x)
        return x
        

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block1 = DecoderBlock(in_channels, 256)    # 10 -> 20
        self.block2 = DecoderBlock(256, 128)    # 20 -> 40
        self.block3 = DecoderBlock(128, 64)     # 40 -> 80
        self.block4 = DecoderBlock(64, 64)       # 80 -> 160
        self.seg = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
        
        self.red_dim1 = nn.Conv2d(256, 64, 1)
        self.red_dim2 = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        x1 = self.block1(x)   # [2, 256, 20, 20]
        x2 = self.block2(x1)  # [2, 128, 40, 40]
        x3 = self.block3(x2)  # [2, 64, 80, 80]
        x4 = self.block4(x3)  # [2, 64, 160, 160]
        x = self.seg(x4)
        x1 = self.red_dim1(x1)
        x2 = self.red_dim2(x2)
        return x1, x2, x3, x4, x


class ResNetVAEEncoder(nn.Module):
    def __init__(self, latent_dims, in_c=4):
        super(ResNetVAEEncoder, self).__init__()
        self.encoder = Encoder(in_c)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dims)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dims)

    def forward(self, x):
        x1, x3 = self.encoder(x)   # [1, 4, 160, 160] -> 
        x = self.adaptive_pool(x3)
        x = x.view(x.size(0), -1)  # Flatten the tensor     [1, 8192]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder1(nn.Module):
    def __init__(self, latent_dims):
        super(ConvDecoder1, self).__init__()
        self.decoder_input = nn.Linear(latent_dims, 64 * 20 * 20)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),    # output: 32 x 45 x 45
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),    # output: 16 x 90 x 90
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1), # output: 4 x 180 x 180
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1),   # output: 4 x 180 x 180
            nn.Tanh()  # Assuming input images are normalized between -1 and 1
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 64, 20, 20)  # Unflatten batch of feature vectors to a batch of multi-channel feature maps
        z = self.deconv_layers(z)
        z = F.interpolate(z, size=(160, 160), mode='bilinear', align_corners=False)  # Resize to match input
        return z


class Rec1(nn.Module):
    def __init__(self, latent_dim):
        super(Rec1, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.decoder = ConvDecoder1(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, s_f, t_f):
        s_f = self.adaptive_pool(s_f)
        s_f = s_f.view(s_f.size(0), -1)  # Flatten the tensor     [1, 8192]
        s_mu = self.fc_mu(s_f)
        s_logvar = self.fc_logvar(s_f)
        s_z = self.reparameterize(s_mu, s_logvar)
        s_rec = self.decoder(s_z)
        
        # 重建目标域图像
        t_f = self.adaptive_pool(t_f)
        t_f = t_f.view(t_f.size(0), -1)
        t_mu = self.fc_mu(t_f)
        t_logvar = self.fc_logvar(t_f)
        t_z = self.reparameterize(t_mu, t_logvar)
        t_rec = self.decoder(t_z)

        return s_rec, t_rec


class ConvDecoder2(nn.Module):
    def __init__(self, latent_dims):
        super(ConvDecoder2, self).__init__()
        self.decoder_input = nn.Linear(latent_dims, 256 * 5 * 5)
        # self.decoder_input = nn.Linear(latent_dims, 256 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),  # output: 128 x 11 x 11
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),   # output: 64 x 22 x 22
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),    # output: 32 x 45 x 45
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),    # output: 16 x 90 x 90
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1), # output: 4 x 180 x 180
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1),   # output: 4 x 180 x 180
            nn.Tanh()  # Assuming input images are normalized between -1 and 1
        )

    def forward(self, z):
        z = self.decoder_input(z)   # [2, 4096]
        z = z.view(-1, 256, 5, 5)   # Unflatten batch of feature vectors to a batch of multi-channel feature maps
        # z = z.view(-1, 256, 4, 4)   # Unflatten batch of feature vectors to a batch of multi-channel feature maps
        z = self.deconv_layers(z)
        z = F.interpolate(z, size=(160, 160), mode='bilinear', align_corners=False)  # Resize to match input
        return z


class Rec2(nn.Module):
    def __init__(self, latent_dim):
        super(Rec2, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.decoder = ConvDecoder2(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, s_f, t_f):
        s_f = self.adaptive_pool(s_f)
        s_f = s_f.view(s_f.size(0), -1)  # Flatten the tensor     [1, 8192]
        s_mu = self.fc_mu(s_f)
        s_logvar = self.fc_logvar(s_f)
        s_z = self.reparameterize(s_mu, s_logvar)
        s_rec = self.decoder(s_z)
        
        # 重建目标域图像
        t_f = self.adaptive_pool(t_f)
        t_f = t_f.view(t_f.size(0), -1)
        t_mu = self.fc_mu(t_f)
        t_logvar = self.fc_logvar(t_f)
        t_z = self.reparameterize(t_mu, t_logvar)
        t_rec = self.decoder(t_z)

        return s_rec, t_rec



if '__main__' == __name__:
    # # P
    # latent_dims = 128  
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ResNetVAE(latent_dims, in_c=4).to(device)

    # input_tensor = torch.randn(1, 4, 160, 160).to(device)

    # reconstructed, _, _ = model(input_tensor)

    # print(reconstructed.shape)
    
    
    model = SegmentationDecoder(512, 3)
    # inp = torch.randn((2, 512, 10, 10))
    # inp2 = torch.randn((2, 512, 10, 10))
    inp = torch.randn((2, 512, 10, 10))
    inp2 = torch.randn((2, 512, 40, 40))
    oup = model(inp)
    print(oup.shape)

    
    