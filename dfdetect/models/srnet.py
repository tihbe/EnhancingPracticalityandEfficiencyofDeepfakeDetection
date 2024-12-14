import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRNetBlockType1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=1,
        act_function=F.relu,
        kernel_size=3,
        norm_type=nn.BatchNorm2d,
    ):
        super(SRNetBlockType1, self).__init__()
        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = norm_type(out_channels)
        self.act_function = act_function

    def forward(self, x):
        return self.act_function(self.norm(self.layer(x)))


class SRNetBlockType2(nn.Module):
    def __init__(self, in_channels, out_channels, act_function=F.relu, kernel_size=3):
        super(SRNetBlockType2, self).__init__()
        self.layer1 = SRNetBlockType1(
            in_channels, out_channels, act_function=act_function
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)

        return torch.add(x, out2)


class SRNetBlockType3(nn.Module):
    def __init__(self, in_channels, out_channels, act_function=F.relu):
        super(SRNetBlockType3, self).__init__()
        self.layer_parrallel = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.layer1 = SRNetBlockType1(
            in_channels, out_channels, act_function=act_function
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.pool(out)
        out = self.layer_parrallel(x) + out
        return out


class SRNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        act_function=F.relu,
        num_type2_layers=5,
        type_2_layer_feat_size=16,
        type_3_layer_sizes=[16, 64, 128, 256],
        nb_features=512,
        type_1_kernel_size=3,
    ):
        super(SRNet, self).__init__()
        self.nb_features = nb_features

        type_2_layers = (
            SRNetBlockType2(
                type_2_layer_feat_size,
                type_2_layer_feat_size,
                act_function=act_function,
            )
            for _ in range(num_type2_layers)
        )

        type_3_layer_sizes.insert(0, type_2_layer_feat_size)
        type_3_layers = (
            SRNetBlockType3(i, j, act_function=act_function)
            for i, j in zip(type_3_layer_sizes[:-1], type_3_layer_sizes[1:])
        )

        self.feat_extractor = nn.Sequential(
            SRNetBlockType1(
                in_channels,
                64,
                act_function=act_function,
                kernel_size=type_1_kernel_size,
            ),
            SRNetBlockType1(
                64,
                type_2_layer_feat_size,
                act_function=act_function,
                kernel_size=type_1_kernel_size,
            ),
            *type_2_layers,
            *type_3_layers,
            SRNetBlockType1(
                type_3_layer_sizes[-1],
                nb_features,
                stride=2,
                padding=0,
                act_function=act_function,
            ),
            nn.Conv2d(
                in_channels=nb_features,
                out_channels=nb_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(nb_features),
        )

        self.fc = nn.Linear(nb_features * 1 * 1, 2)

    def forward(self, inputs):
        feat = self.feat_extractor(inputs)
        feat = torch.mean(feat, dim=(2, 3), keepdim=True)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out
