from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import timm
import torch.nn.functional as F



def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.double_up =nn.Upsample(
                scale_factor=4, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):

       
        if self.bilinear:
            x1 = self.conv1x1(x1)
        
        if x1.size(3) != x2.size(3):
          x1 = self.up(x1)
         
        elif x1.size(3) == x2.size(3):
            x1 = self.double_up(x1)
            x2 = self.double_up(x2)

        
        if x1.size(2) != x2.size(2):
            h1, h2 = x1.size(2), x2.size(2)
            if h1 > h2:
            # 裁剪 x1 至 h2 的高度
                 x1 = x1[:, :, :h2, :]
            else:
                if h1 == 20:
                # 计算需要填充的高度
                    pad_needed = h2 - h1
                    # 在高度维度（第三维）的下方填充 pad_needed 个像素
                    x1 = F.pad(x1, (0, 0, 0, pad_needed))
                else:
                # 裁剪 x2 至 h1 的高度
                    x2 = x2[:, :, :h1, :]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
       
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                #   'feature_chns': [16, 32, 64, 128, 256], ###unet_fugc_unimatch_labeled_7
                     'feature_chns': [16, 32, 96, 256, 384], ###unet_fugc_unimatch_labeled_50_moda

                #   'feature_chns': [32, 64, 128, 256, 512], ###unet_fugc_unimatch_labeled_7_3 ###对比unet增加通道数的影响
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)
        
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)
        
        output = self.decoder(feature)
        return output



##=========================================my-unet-2d=========================================================


def encode_for_convnext(e, x,):
    encode = []
    x = e.stem_0(x)
    x = e.stem_1(x)
 
    encode.append(x)

    x = e.stages_0(x)

    encode.append(x)

    x = e.stages_1(x)
  
    encode.append(x)

    x=e.stages_2(x)
  
    encode.append(x)

    x = e.stages_3(x)   
    
    encode.append(x)

    # for  fea in encode:
    #     print(f'fea.shape:{fea.shape}')


    

    
    return encode
def encode_for_pvt(e, x):

    encode=[]
    x = e.patch_embed(x)
    x_ = x.permute(0,3,1,2).contiguous()
    encode.append(x_)
    x = x_.permute(0,2,3,1).contiguous()
  
    x = e.stages_0(x)

    encode.append(x)

    x = e.stages_1(x)
    
    encode.append(x)

    x= e.stages_2(x)
   
    encode.append(x)

    x= e.stages_3(x)
 
    encode.append(x)

    return encode


class encoder2d_unet(nn.Module):
    def __init__(self, model_name, model_path, in_chns, class_num):
        super(encoder2d_unet, self).__init__()

        self.encoder_dim = {
            'pvt_v2_b1': [64, 64, 128, 320, 512],
            'pvt_v2_b2': [64, 64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }

        # 检查模型名称是否在 self.encoder_dim 的键中
        if model_name not in self.encoder_dim:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models are: {list(self.encoder_dim.keys())}")

        self.arch = model_name

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=False, in_chans=in_chns, num_classes=0, global_pool='', features_only=True,
        )

        # 根据 model_path 加载对应的模型预训练权重
        if model_path:
            state_dict = torch.load(model_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)

        params = {'in_chns': in_chns,
                  'feature_chns': self.encoder_dim[self.arch],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.decoder = Decoder(params)
        print(f'self.arch: {self.arch}')

    def forward(self, x, need_fp=False):
        feature = encode_for_pvt(self.encoder, x)
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)
        
        output = self.decoder(feature)
        return output

    
if __name__=='__main__':

     x = torch.rand(1,3,336,544)
    #  net = UNet(3,3)
    #  out = net(x)
    #  print(f'out.shape:{out.shape}')



     net = encoder2d_unet(3,3)
     out = net(x)
     print(f'out.shape:{out.shape}')
