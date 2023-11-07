# -*- coding: utf-8 -*-
"""
Module: cnn_model_factory.py
Author: Daniel Tebor
Description: This module contains a factory class for creating or loading CNN models using PyTorch.
"""

import logging
from enum import Enum
from typing import Union

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
    """
    A factory class for creating CNN models using PyTorch.

    Attributes:
        None

    Methods:
        _create_alexnet(num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
            Creates an AlexNet model with the specified number of output classes.

        _create_resnet50(num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
            Creates a ResNet50 model with the specified number of output classes.

        _create_resnet101(num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
            Creates a ResNet101 model with the specified number of output classes.

        _create_resnet152(num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
            Creates a ResNet152 model with the specified number of output classes.

        create_model(model_name: Union[str, ModelName], num_classes: int) -> (nn.Module, nn.Module, optim.Optimizer):
            Creates a CNN model based on the specified model name and number of classes.

        load_model(model_name: Union[str, ModelName], model_file: str) -> (nn.Module, nn.Module, optim.Optimizer):
            Loads a saved model from a file and returns the model, criterion, and optimizer.

        get_model_input_shape(model_name: Union[str, ModelName]) -> (int, int, int):
            Gets the input shape for the specified model.
    """

    def _create_alexnet(self, num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
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
    
    def _create_resnet50(self, num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
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
    
    def _create_resnet101(self, num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
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
    
    def _create_resnet152(self, num_classes: int = None) -> (nn.Module, nn.Module, optim.Optimizer):
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
        
    def create_model(self, model_name: Union[str, ModelName], num_classes: int) -> (nn.Module, nn.Module, optim.Optimizer):
        """
        Creates a PyTorch model based on the specified model name and number of classes.

        Args:
            model_name (Union[str, ModelName]): The name of the model to create.
            num_classes (int): The number of output classes for the model.

        Returns:
            Tuple containing the PyTorch model, loss function, and optimizer.
        """

        if num_classes == None:
            raise RuntimeError('Num classes unspecified')

        # Convert model_name to ModelName enum if it is a string.
        if isinstance(model_name, str):
            for name in ModelName:
                if model_name.lower() == name.value.lower():
                    model_name = name
                    break
            else:
                raise RuntimeError(f'Unknown model: {model_name}')
            
        model = None
        criterion = None
        optimizer = None
        if model_name == ModelName.ALEXNET:
            model, criterion, optimizer = self._create_alexnet(num_classes)
        elif model_name == ModelName.RESNET50:
            model, criterion, optimizer = self._create_resnet50(num_classes)
        elif model_name == ModelName.RESNET101:
            model, criterion, optimizer = self._create_resnet101(num_classes)
        elif model_name == ModelName.RESNET152:
            model, criterion, optimizer = self._create_resnet152(num_classes)
        
        log = logging.getLogger(__name__)
        log.info(f'Created {model_name.value} model with {num_classes} output classes')
        return model, criterion, optimizer
        
    def load_model(self, model_filename: str) -> (nn.Module, nn.Module, optim.Optimizer):
        """
        Loads a saved model from a pt file and returns the model, criterion, and optimizer.

        Args:
            model_name (Union[str, ModelName]): The name of the model to create.
            model_path (str): The path to the saved model file.

        Returns:
            Tuple containing the loaded model, criterion, and optimizer.
        """

        model_name = model_filename.split('_')[0]

        output_reader = OutputReader()
        state_dict = output_reader.read_model(model_filename)

        # Get number of classes from saved model.
        num_classes = state_dict['fc.weight'].shape[0]

        model, criterion, optimizer = self.create_model(model_name=model_name, num_classes=num_classes)
        model.load_state_dict(state_dict=state_dict)

        return model, criterion, optimizer

    def get_model_input_shape(self, model_name: Union[str, ModelName]) -> (int, int, int):
        """
        Returns the input shape of the specified model.

        Args:
            model_name (Union[str, ModelName]): The name of the model.

        Returns:
            Tuple[int, int, int]: The input shape of the model.
        """

        if isinstance(model_name, str):
            try:
                model_name = ModelName[model_name.upper()]
            except KeyError:
                raise RuntimeError(f'Unknown model name: {model_name}')

        if model_name == ModelName.ALEXNET:
            return 3, 256, 256
        elif model_name == ModelName.RESNET50:
            return 3, 224, 224
        elif model_name == ModelName.RESNET101:
            return 3, 224, 224
        elif model_name == ModelName.RESNET152:
            return 3, 224, 224
        else:
            raise RuntimeError(f'Unknown model name: {model_name}')