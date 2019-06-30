import torch
from torch import nn
import torch.nn.functional as F

from modules.conv import conv, conv_dw


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
            conv(num_channels, num_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return [heatmaps]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class UShapedContextBlock(nn.Module):
    def __init__(self, in_channels, to_onnx=False):
        super().__init__()
        self.to_onnx = to_onnx
        self.encoder1 = nn.Sequential(
            conv(in_channels, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.encoder2 = nn.Sequential(
            conv(in_channels*2, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder2 = nn.Sequential(
            conv(in_channels*2 + in_channels*2, in_channels*2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder1 = nn.Sequential(
            conv(in_channels*3, in_channels*2),
            conv(in_channels*2, in_channels)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        size_e1 = (e1.size()[2], e1.size()[3])
        size_x = (x.size()[2], x.size()[3])
        if self.to_onnx:  # Need interpolation to fixed size for conversion
            size_e1 = (16, 16)
            size_x = (32, 32)
        d2 = self.decoder2(torch.cat([e1, F.interpolate(e2, size=size_e1,
                                                        mode='bilinear', align_corners=False)], 1))
        d1 = self.decoder1(torch.cat([x, F.interpolate(d2, size=size_x,
                                                       mode='bilinear', align_corners=False)], 1))

        return d1


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, to_onnx):
        super().__init__()

        self.trunk = nn.Sequential(
            UShapedContextBlock(in_channels, to_onnx),
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return [heatmaps]


class SinglePersonPoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=17, to_onnx=False):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),  # conv5_5
        )
        self.cpm = nn.Sequential(
            conv(512, 256),
            conv(256, 128),
        )

        self.initial_stage = InitialStage(num_channels, num_heatmaps)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps, num_channels, num_heatmaps,
                                                          to_onnx))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-1]], dim=1)))

        return stages_output
