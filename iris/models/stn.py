from kornia.geometry.transform import warp_perspective
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Stn(nn.Module):
    def __init__(self):
        super(Stn, self).__init__()

        self.localization = models.mobilenet_v3_small().features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(576 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Linear(32, 8),
            nn.Tanh(),
        )
        
        with torch.no_grad():
            self.fc[2].weight.data.zero_()
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = self.avgpool(xs)
        xs = xs.view(x.size(0), -1)
        
        theta = self.fc(xs)
        ones = torch.ones(theta.size(0), 1, dtype=theta.dtype, device=theta.device)
        theta = torch.cat([theta, ones], dim=1)
        theta[:, 0] = theta[:, 0] + 1.0 # Scale x, range [0, 2]
        theta[:, 4] = theta[:, 4] + 1.0 # Scale y, range [0, 2]
        theta[:, 1] = theta[:, 1] # Shear x, range [-1, 1]
        theta[:, 3] = theta[:, 3] # Shear y, range [-1, 1]
        theta[:, 2] = theta[:, 2] * 0.5 # Translation x, range [-0.5, 0.5]
        theta[:, 5] = theta[:, 5] * 0.5 # Translation y, range [-0.5, 0.5]
        theta[:, 6] = theta[:, 6] * 0.001 # Perspective x, range [-0.001, 0.001]
        theta[:, 7] = theta[:, 7] * 0.001 # Perspective y, range [-0.001, 0.001]
        theta = theta.view(-1, 3, 3)
        
        return warp_perspective(x, theta, dsize=(x.size(2), x.size(3)), align_corners=True)