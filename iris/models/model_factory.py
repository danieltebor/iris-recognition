import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from iris.models.stn_convnext_tiny import StnConvNextTiny


class ModelConfig:
    def __init__(self,
                 model_name: str,
                 num_classes: bool = None,
                 use_pretrained: bool = False):
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        
    def __str__(self):
        return f'ModelConfig(model_name={self.model_name}, num_classes={self.num_classes}, use_pretrained={self.use_pretrained})'

class ModelFactory:
    model_input_shape = {
        'convnext_tiny': (3, 224, 224),
        'convnext_small': (3, 224, 224),
        'resnet50': (3, 224, 224),
        'stn_convnext_tiny': (3, 224, 224),
    }
    model_normalization = {
        'convnext_tiny': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        'convnext_small': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        'resnet50': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        'stn_convnext_tiny': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    }
    
    @classmethod
    def create_classifier(cls, model_config: ModelConfig) -> nn.Module:
        valid_model_names = cls.model_input_shape.keys()
        assert model_config.model_name in valid_model_names, f'Model name must be one of: {valid_model_names}'
        assert model_config.num_classes is not None, 'Number of classes must be specified for classifier'

        model = None
        
        if model_config.model_name == 'convnext_tiny':
            if model_config.use_pretrained:
                model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            else:
                model = models.convnext_tiny()
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, model_config.num_classes)
        elif model_config.model_name == 'convnext_small':
            if model_config.use_pretrained:
                model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            else:
                model = models.convnext_small()
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, model_config.num_classes)
        elif model_config.model_name == 'resnet50':
            if model_config.use_pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50() 
            model.fc = nn.Linear(model.fc.in_features, model_config.num_classes)
        elif model_config.model_name == 'stn_convnext_tiny':
            model = StnConvNextTiny(num_classes=model_config.num_classes, use_pretrained_convnext=model_config.use_pretrained)
        
        return model

    @classmethod
    def create_feature_extractor(cls, model_config: ModelConfig) -> nn.Module:
        valid_model_names = cls.model_input_shape.keys()
        assert model_config.model_name in valid_model_names, f'Model name must be one of: {valid_model_names}'
        
        model = None
        
        if model_config.model_name == 'convnext_tiny':
            if model_config.use_pretrained:
                model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            else:
                model = models.convnext_tiny()
            model.classifier[-1] = nn.Identity()
        elif model_config.model_name == 'convnext_small':
            if model_config.use_pretrained:
                model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            else:
                model = models.convnext_small()
            model.classifier[-1] = nn.Identity()
        elif model_config.model_name == 'resnet50':
            if model_config.use_pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50()
            model.fc = nn.Identity()
        elif model_config.model_name == 'stn_convnext_tiny':
            model = StnConvNextTiny(use_pretrained_convnext=model_config.use_pretrained)
            model.fc = nn.Identity()
        
        return model
    
    @classmethod
    def get_input_shape(cls, model_name: str) -> tuple[int, int, int]:
        return cls.model_input_shape[model_name]
    
    @classmethod
    def get_normalization_transform(cls, mode_name: str) -> transforms.Normalize:
        return cls.model_normalization[mode_name]
    
    @classmethod
    def get_resize_transform(cls, model_name: str) -> transforms.Resize:
        return transforms.Resize(cls.model_input_shape[model_name][1:])