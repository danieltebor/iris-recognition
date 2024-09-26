# -*- coding: utf-8 -*-
"""
Module: cnn_model_operator_factory.py
Author: Daniel Tebor
Description: This module contains a factory class for creating model operators for different things.
"""

from enum import Enum
from typing import  Tuple, Union

from torch import nn, optim

from models.model_operators.cnn_model_operator import CNNModelOperator
from models.model_operators.cnn_merged_rgb_model_operator import CNNMergedRGBModelOperator
from models.model_operators.cnn_wide_input_model_operator import CNNWideInputModelOperator
from models.model_operators.cnn_triplet_loss_model_operator import CNNTripletLossModelOperator
from models.model_operators.cnn_triplet_loss_wide_input_model_operator import CNNTripletLossWideInputModelOperator


class ModelOperatorName(Enum):
    CNN = 'cnn'
    MERGED_RGB = 'merged_rgb'
    WIDE_INPUT = 'wide_input'
    TRIPLET_LOSS = 'triplet_loss'
    TRIPLET_LOSS_WIDE_INPUT = 'triplet_loss_wide_input'

class CNNModelOperatorFactory:
    def __init__(self,
                 model: nn.Module,
                 modelname: str,
                 num_output_classes: int,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer):
        self._model = model
        self._modelname = modelname
        self._num_output_classes = num_output_classes
        self._criterion = criterion
        self._optimizer = optimizer

    def _create_base(self) -> CNNModelOperator:
        return CNNModelOperator(model=self._model,
                                modelname=self._modelname,
                                num_output_classes=self._num_output_classes,
                                criterion=self._criterion,
                                optimizer=self._optimizer)

    def _create_merged_rgb(self) -> CNNModelOperator:
        return CNNMergedRGBModelOperator(model=self._model,
                                         modelname=self._modelname,
                                         num_output_classes=self._num_output_classes,
                                         criterion=self._criterion,
                                         optimizer=self._optimizer)

    def _create_wide_input(self) -> CNNModelOperator:
        return CNNWideInputModelOperator(model=self._model,
                                         modelname=self._modelname,
                                         num_output_classes=self._num_output_classes,
                                         criterion=self._criterion,
                                         optimizer=self._optimizer)
    
    def _create_triplet_loss(self) -> CNNModelOperator:
        return CNNTripletLossModelOperator(model=self._model,
                                           modelname=self._modelname,
                                           num_output_classes=self._num_output_classes,
                                           criterion=self._criterion,
                                           optimizer=self._optimizer)

    def _create_triplet_loss_wide_input(self) -> CNNModelOperator:
        return CNNTripletLossWideInputModelOperator(model=self._model,
                                                    modelname=self._modelname,
                                                    num_output_classes=self._num_output_classes,
                                                    criterion=self._criterion,
                                                    optimizer=self._optimizer)

    def create_model_operator(self, model_name: Union[str, ModelOperatorName]):
        if model_name == ModelOperatorName.CNN:
            return self._create_base()
        elif model_name == ModelOperatorName.MERGED_RGB:
            return self._create_merged_rgb()
        elif model_name == ModelOperatorName.WIDE_INPUT:
            return self._create_wide_input()
        elif model_name == ModelOperatorName.TRIPLET_LOSS:
            return self._create_triplet_loss()
        elif model_name == ModelOperatorName.TRIPLET_LOSS_WIDE_INPUT:
            return self._create_triplet_loss_wide_input()
        else:
            raise ValueError(f'Invalid model operator name: {model_name}')