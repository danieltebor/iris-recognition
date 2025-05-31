import torch.nn as nn
import torchvision.models as models

from iris.models.stn import Stn


class StnConvNextTiny(nn.Module):
    def __init__(self, num_classes: int = None, use_pretrained_convnext: bool = False):
        super(StnConvNextTiny, self).__init__()

        self.stn = Stn()
        
        if use_pretrained_convnext:
            self.convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.convnext_tiny = models.convnext_tiny()
        
        if num_classes is not None:
            self.fc = nn.Linear(self.convnext_tiny.classifier[-1].in_features, num_classes)
        else:
            self.fc = self.convnext_tiny.classifier[-1]
            
        self.convnext_tiny.classifier[-1] = nn.Identity()
        
    def forward(self, x):
        x = self.stn(x)
        x = self.convnext_tiny(x)
        return self.fc(x)