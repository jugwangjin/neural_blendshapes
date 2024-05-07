import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi * 1.1
HALF_PI = 1.1 * torch.pi / 2


def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()


class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super(ResnetEncoder, self).__init__()
        try:
            self.encoder = torchvision.models.resnet34(weights='DEFAULT')
        except:
            self.encoder = torchvision.models.resnet34(pretrained=True)

        self.outsize = outsize

        self.encoder = torch.nn.Sequential(*list(self.encoder.children()))

        feature_size = self.encoder[-1].in_features
        self.encoder = self.encoder[:-1]

        self.feature_size = feature_size
        
        self.layers = nn.Sequential(
            nn.Linear(feature_size + 68*3, min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), outsize + 68*3),
        )

        self.resize = nn.Upsample(size=(224, 224), mode='bilinear')
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        initialize_weights(self.layers, gain=0.01)

    def forward(self, image, lmks):
        # print(inputs.shape)
        # with torch.no_grad():
        inputs = self.resize(image)
        # normalize inputs,  mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        inputs = (inputs - self.mean) / self.std
         
        features = self.encoder(inputs)
        features = torch.cat([features.view(-1, self.feature_size), lmks[..., :3].reshape(lmks.size(0), -1)], dim=-1)
        # features = torch.cat([features, lmks[..., :3].reshape(lmks.size(0), -1)], dim=-1)
        features = self.layers(features)

        features, estim_landmarks = features[..., :-68*3], features[..., -68*3:]

        # we will use first 52 elements as FACS features
        features[..., :56] = torch.nn.functional.sigmoid(features[..., :56])

        features[..., 53:56] = features[..., 53:56] * PI - HALF_PI
        # features[..., -4:-1] = 0
        features[..., -2] = 0
        features[..., -1] = torch.sigmoid(features[..., -1]) * 2

        return features, estim_landmarks

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



