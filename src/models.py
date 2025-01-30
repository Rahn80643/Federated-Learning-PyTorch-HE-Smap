#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import torch
from torch import nn
import torchvision.models as tvmodels
from torchvision.transforms import Resize

from ghostnet import ghostnet as load_ghostnet
#from tinynet import tinynet as load_tinynet
from models_utils import *

from collections import OrderedDict

def replace_hardsigmoid_with_sigmoid(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(module, name, nn.Sigmoid())
        else:
            replace_hardsigmoid_with_sigmoid(child)


# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class mlp_mnist(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mlp_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels*28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class cnn_mnist(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(cnn_mnist, self).__init__()

        self.resize = Resize((28, 28))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ported from 2016 TensorFlow CIFAR-10 tutorial)
class cnn_cifar10(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(cnn_cifar10, self).__init__()

        self.resize = Resize((24, 24))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Gradient-Based Learning Applied to Document Recognition"
class lenet5_orig(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(lenet5_orig, self).__init__()
        orig_activation = True
        orig_norm = True
        orig_s = True
        orig_c3 = True
        orig_f7 = True

        activation = nn.Tanh if orig_activation else nn.ReLU
        activation_constant = 1.7159 if orig_activation else 1
        norm = nn.BatchNorm2d if orig_norm else nn.Identity
        c1 = nn.Conv2d(num_channels, 6, 5)
        s2 = LeNet5_Orig_S(6) if orig_s else nn.MaxPool2d(2, 2)
        c3 = LeNet5_Orig_C3() if orig_c3 else nn.Conv2d(6, 16, 5)
        s4 = LeNet5_Orig_S(16) if orig_s else nn.MaxPool2d(2, 2)
        c5 = nn.Conv2d(16, 120, 5, bias=True)
        f6 = nn.Linear(120, 84)
        f7 = LeNet5_Orig_F7(84, 10) if orig_f7 else nn.Linear(84, 10)

        self.resize = Resize((32, 32))

        self.feature_extractor = nn.Sequential(
            c1,
            norm(6),
            activation(), Multiply(activation_constant),
            s2,
            c3,
            norm(16),
            activation(), Multiply(activation_constant),
            s4,
            c5,
            norm(120),
            activation(), Multiply(activation_constant),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            f6,
            activation(), Multiply(activation_constant),
            f7,
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ported from 2016 TensorFlow CIFAR-10 tutorial)
#     * LocalResponseNorm replaced with BatchNorm2d/GroupNorm/Identity
#     * Normalization placed always before ReLU
#     * Conv2d-Normalization-ReLU optionally replaced by GhostModule from "GhostNet: More Features from Cheap Operations"
class lenet5(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(lenet5, self).__init__()

        norm = model_args['norm'] if 'norm' in model_args else 'batch'
        if norm == 'batch':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(64)
        elif norm == 'group':
            # Group Normalization paper suggests 16 channels per group is best
            norm1 = nn.GroupNorm(int(64/16), 64)
            norm2 = nn.GroupNorm(int(64/16), 64)
        elif norm == None:
            norm1 = nn.Identity(64)
            norm2 = nn.Identity(64)
        else:
            raise ValueError("Unsupported norm '%s' for LeNet5")
        if 'ghost' in model_args and model_args['ghost']:
            block1 = GhostModule(num_channels, 64, 5, padding='same', norm=norm)
            block2 = GhostModule(64, 64, 5, padding='same', norm=norm)
        else:
            block1 = nn.Sequential(
                nn.Conv2d(num_channels, 64, 5, padding='same'),
                norm1,
                nn.ReLU(),
            )
            block2 = nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                norm2,
                nn.ReLU(),
            )

        self.resize = Resize((24, 24))

        self.feature_extractor = nn.Sequential(
            block1,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
            block2,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 384), # 5*5 if input is 32x32
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes))

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class CompactNetFinalAdjusted(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(CompactNetFinalAdjusted, self).__init__()
        
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # First block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(48 * 3 * 3, 120), 

            # cifar100, 32x 32 -> 64 features; tinyImagenet 64x 64 -> 64*4*4 features
            # nn.Linear(64*4*4, 768), 
            nn.Linear(64, 768), 
            nn.ReLU(inplace=True),
            nn.Linear(768, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 20241028, Vit
class ViT(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(ViT, self).__init__()
        pretrained = True # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'vit_b_16')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        # change the number of output classes
        self.model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
    

# 20240910, for DLG attack
class lenet_DLG(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(lenet_DLG, self).__init__()

        self.resize = Resize((32, 32))
        # self.resize = Resize((32, 32), interpolation=0) # Brutal adjust for FedML-HE
        # self.resize = Resize((224, 224))

        act = nn.ReLU
        self.body = nn.Sequential(
            # nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes),
            #act(),
            #nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.resize(x)
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        #print(feature.size())
        out = self.fc(feature)
        return out 

class AlexNet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(AlexNet, self).__init__()
        
        pretrained = True
        freeze = False
        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'alexnet')(pretrained=True)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, out_features=num_classes)
        stop =1
        

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x    
    
class ShuffleNet_V2_X0_5(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(ShuffleNet_V2_X0_5, self).__init__()
        
        pretrained = True
        freeze = False
        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'shufflenet_v2_x0_5')(pretrained=True)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features=num_classes)
        stop =1
        

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x   
    
# 2024.09.17, from inverting gradients; TODO: can't be model.summary, dimension is incompatible
class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, num_classes, num_channels, model_args):
        width=32 
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            # ('linear', torch.nn.Linear(36 * width, num_classes))
            ('linear', torch.nn.Linear(512, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)

class mnasnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mnasnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        if pretrained:
            if width == 1:
                self.model = tvmodels.mnasnet1_0(pretrained=True, dropout=dropout)
            elif width == 0.5:
                self.model = tvmodels.mnasnet0_5(pretrained=True, dropout=dropout)
            elif width == 0.75:
                self.model = tvmodels.mnasnet0_75(pretrained=True, dropout=dropout)
            elif width == 1.3:
                self.model = tvmodels.mnasnet1_3(pretrained=True, dropout=dropout)
            else:
                raise ValueError('Unsupported width for pretrained MNASNet: %s' % width)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            self.model = tvmodels.mnasnet.MNASNet(alpha=width, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

class ghostnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(ghostnet, self).__init__()
        width = model_args['width'] if 'width' in model_args else 1.0
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)
        #self.resize = Resize(24)

        if pretrained:
            if width != 1:
                raise ValueError('Unsupported width for pretrained GhostNet: %s' % width)
            self.model = load_ghostnet(width=1, dropout=dropout)
            self.model.load_state_dict(torch.load('models/ghostnet.pth'), strict=True)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        else:
            self.model = load_ghostnet(num_classes=num_classes, width=width, dropout=dropout)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x


# since pytorch doesn't offer mobilenet v1 by default
class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class mobilenet_v1(nn.Module):
    # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenet.py

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """
    
    def __init__(self, num_classes, num_channels, model_args): # for CIFAR 100
       super().__init__()

       self.resize = Resize(224)
       width_multiplier=1
       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       self.fc = nn.Linear(int(1024 * alpha), num_classes)
    #    self.fc = nn.Linear(int(1024 * alpha), 1000)
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.resize(x)
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# 2024.06.24: new added: mobilenetv2
class mobilenet_v2(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mobilenet_v2, self).__init__()
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'mobilenet_v2')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, out_features=num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(mobilenet_v3, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'small'
        # variant = model_args['variant'] if 'variant' in model_args else 'large'
        pretrained = True # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'mobilenet_v3_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, self.model.classifier[0].out_features)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

        replace_hardsigmoid_with_sigmoid(self.model)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

class efficientnetb0(nn.Module): # add b5, b7
    def __init__(self, num_classes, num_channels, model_args):
        super(efficientnetb0, self).__init__()
        variant = 'b0' 
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
    
class efficientnetb5(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(efficientnetb5, self).__init__()
        variant = 'b5' 
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
    
class efficientnetb7(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(efficientnetb7, self).__init__()
        variant = 'b7' 
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

# TODO: , ViT
# new: 2024.06.24
class resnet18(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(resnet18, self).__init__()
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'resnet18')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x


class resnet34(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(resnet34, self).__init__()
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'resnet34')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
    

class resnet50(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(resnet50, self).__init__()
        # pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        pretrained = True
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'resnet50')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
    
class vit(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(vit, self).__init__()
        variant = model_args['variant'] if 'variant' in model_args else 'b_16'
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'vit_{variant}')(pretrained=pretrained)
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        # self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        self.model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True) # 768

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x