import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from nets.layers import FPN, SSH
import numpy as np


class PerimetryNet(nn.Module):
    def __init__(self, cfg=None, pretrained=False, mode='train'):  
        super(PerimetryNet, self).__init__()

        backbone = models.resnet50(pretrained=pretrained)       
        self.body = _utils.IntermediateLayerGetter(backbone, cfg[
               'return_layers'])         

        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]  
        self.fpn = FPN(in_channels_list, cfg['out_channel']) 
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel']) 
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])
        

        self.fc_yaw_class1 = nn.Linear(cfg['out_channel'] * 56 * 56, 28) 
        self.fc_yaw_class2 = nn.Linear(cfg['out_channel'] * 28 * 28, 28)
        self.fc_yaw_class3 = nn.Linear(cfg['out_channel'] * 14 * 14, 28)

        self.fc_regre1 = nn.Linear(cfg['out_channel'] * 56 * 56, 2) 
        self.fc_regre2 = nn.Linear(cfg['out_channel'] * 28 * 28, 2)
        self.fc_regre3 = nn.Linear(cfg['out_channel'] * 14 * 14, 2)

        self.fc_pitch_class1 = nn.Linear(cfg['out_channel'] * 56 * 56, 28)  
        self.fc_pitch_class2 = nn.Linear(cfg['out_channel'] * 28 * 28, 28)
        self.fc_pitch_class3 = nn.Linear(cfg['out_channel'] * 14 * 14, 28)




    def forward(self, inputs):

        out = self.body.forward(inputs)  

        fpn = self.fpn.forward(out)  

        feature1 = self.ssh1(fpn[0])  
        feature2 = self.ssh2(fpn[1])  
        feature3 = self.ssh3(fpn[2])  

        feature1 = feature1.view(feature1.size(0), -1)  
        feature2 = feature2.view(feature2.size(0), -1)
        feature3 = feature3.view(feature3.size(0), -1)
        
        yaw_class1 = self.fc_yaw_class1(feature1)
        yaw_class2 = self.fc_yaw_class2(feature2)
        yaw_class3 = self.fc_yaw_class3(feature3)
        pitch_class1 = self.fc_pitch_class1(feature1)
        pitch_class2 = self.fc_pitch_class2(feature2)
        pitch_class3 = self.fc_pitch_class3(feature3)
        gaze_regre1 = self.fc_regre1(feature1)
        gaze_regre2 = self.fc_regre2(feature2)
        gaze_regre3 = self.fc_regre3(feature3)

        yaw_class = yaw_class1 + yaw_class2 + yaw_class3
        pitch_class = pitch_class1 + pitch_class2 + pitch_class3
        gaze_regre = gaze_regre1 + gaze_regre2 + gaze_regre3

        return yaw_class, pitch_class, gaze_regre  
