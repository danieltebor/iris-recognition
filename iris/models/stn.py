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
            nn.GELU(),
            nn.Linear(32, 6),
        )
        
        with torch.no_grad():
            self.fc[-1].weight.data.zero_()
            self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = self.avgpool(xs)
        xs = xs.view(x.size(0), -1)
        
        theta = self.fc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)