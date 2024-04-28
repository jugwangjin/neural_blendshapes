import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi * 1.5
HALF_PI = 1.5 * torch.pi / 2

class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super(ResnetEncoder, self).__init__()
        try:
            self.encoder = torchvision.models.resnet34(weights='DEFAULT')
        except:
            self.encoder = torchvision.models.resnet34(pretrained=True)

        self.encoder = torch.nn.Sequential(*list(self.encoder.children()))

        feature_size = self.encoder[-1].in_features
        self.encoder = self.encoder[:-1]
        
        self.layers = nn.Sequential(
            nn.Linear(feature_size, min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), min(feature_size, 512)),
            nn.SiLU(),
            nn.Linear(min(feature_size, 512), outsize),
        )

        self.resize = nn.Upsample(size=(224, 224), mode='bilinear')
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inputs):
        # print(inputs.shape)
        # with torch.no_grad():
        inputs = self.resize(inputs)
        # normalize inputs,  mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        inputs = (inputs - self.mean) / self.std
         
        features = self.encoder(inputs)
        features = features.reshape(features.size(0), -1)
        features = self.layers(features)

        # we will use first 52 elements as FACS features
        features[..., :56] = torch.nn.functional.sigmoid(features[..., :56])

        features[..., 53:56] = features[..., 53:56] * PI - HALF_PI

        features[..., -1] = torch.exp(features[..., -1] * 0.25)

        return features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



