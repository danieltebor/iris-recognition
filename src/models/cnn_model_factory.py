# -*- coding: utf-8 -*-
"""
Module: cnn_model_factory.py
Author: Daniel Tebor
Description: This module contains a factory class for creating or loading CNN models using PyTorch.
"""

import logging
from enum import Enum
from typing import  Tuple, Union

from torch import nn, optim
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

from common.output_reader import OutputReader


class ModelName(Enum):
    ALEXNET = 'alexnet'
    RESNET50 = 'resnet50'
    RESNET101= 'resnet101'
    RESNET152 = 'resnet152'

class CNNModelFactory:
    def _create_alexnet(self, num_classes: int = None) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        if num_classes == None:
            raise RuntimeError("Num classes unspecified")
        
        model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)
            
        if num_classes != None:
            # Resize output layer to match num_classes.
            model.classifier[6] = nn.Linear(4096, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr = 0.001,
                              momentum = 0.9)
        
        return model, criterion, optimizer
    
    def _create_resnet50(self, num_classes: int = None) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        if num_classes == None:
            raise RuntimeError("Num classes unspecified")
        
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        
        if num_classes != None:
            # Resize output layer to match num_classes.
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=0.00002)
        
        return model, criterion, optimizer
    
    def _create_resnet101(self, num_classes: int = None) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        if num_classes == None:
            raise RuntimeError("Num classes unspecified")
        
        model = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)
        
        if num_classes != None:
            # Resize output layer to match num_classes.
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr = 0.01,
                              momentum = 0.9,
                              weight_decay=5e-4)
        
        return model, criterion, optimizer
    
    def _create_resnet152(self, num_classes: int = None) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        if num_classes == None:
            raise RuntimeError("Num classes unspecified")
        
        model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
        
        if num_classes != None:
            # Resize output layer to match num_classes.
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr = 0.01,
                              momentum = 0.9,
                              weight_decay=5e-4)
        
        return model, criterion, optimizer
        
    def create_model(self, modelname: Union[str, ModelName], num_classes: int) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        if num_classes == None:
            raise RuntimeError('Num classes unspecified')

        # Convert model_name to ModelName enum if it is a string.
        if isinstance(modelname, str):
            for name in ModelName:
                if modelname.lower() == name.value.lower():
                    modelname = name
                    break
            else:
                raise RuntimeError(f'Unknown model: {modelname}')
            
        model = None
        criterion = None
        optimizer = None
        if modelname == ModelName.ALEXNET:
            model, criterion, optimizer = self._create_alexnet(num_classes)
        elif modelname == ModelName.RESNET50:
            model, criterion, optimizer = self._create_resnet50(num_classes)
        elif modelname == ModelName.RESNET101:
            model, criterion, optimizer = self._create_resnet101(num_classes)
        elif modelname == ModelName.RESNET152:
            model, criterion, optimizer = self._create_resnet152(num_classes)
        
        log = logging.getLogger(__name__)
        log.info(f'Created {modelname.value} model with {num_classes} output classes')
        return model, criterion, optimizer
        
    def load_model(self, model_filename: str, is_classifier: bool = True) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        modelname = model_filename.split('_')[0]

        output_reader = OutputReader()
        state_dict = output_reader.read_model(model_filename)

        # Get number of classes from saved model.
        if is_classifier:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            num_classes = 1

        model, criterion, optimizer = self.create_model(modelname=modelname, num_classes=num_classes)

        if not is_classifier:
            model = nn.Sequential(*list(model.children())[:-1])

        model.load_state_dict(state_dict=state_dict)

        return model, criterion, optimizer

    def get_model_input_shape(self, modelname: Union[str, ModelName]) -> Tuple[int, int, int]:
        if isinstance(modelname, str):
            try:
                modelname = ModelName[modelname.upper()]
            except KeyError:
                raise RuntimeError(f'Unknown model name: {modelname}')

        if modelname == ModelName.ALEXNET:
            return 3, 256, 256
        elif modelname == ModelName.RESNET50:
            return 3, 224, 224
        elif modelname == ModelName.RESNET101:
            return 3, 224, 224
        elif modelname == ModelName.RESNET152:
            return 3, 224, 224
        else:
            raise RuntimeError(f'Unknown model name: {modelname}')