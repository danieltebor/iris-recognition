# -*- coding: utf-8 -*-
"""
Module: cnn_merged_rgb_model_operator.py
Author: Daniel Tebor
Description: This module contains a class for training and evaluating the ResNet50 model with multi-image input acheived by putting different images in different rgb channels.
"""

import copy
import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms import Grayscale

from common.output_writer import OutputWriter
from models.model_operators.cnn_model_operator import CNNModelOperator


class CNNMergedRGBModelOperator(CNNModelOperator):
    def __init__(self,
                 model: nn.Module,
                 modelname: str,
                 num_output_classes: int,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer):
        super().__init__(model, modelname, num_output_classes, criterion, optimizer)

    def train(self, training_dataloader: DataLoader, testing_dataloader: DataLoader) -> Tuple[int, int]:
        log = logging.getLogger(__name__)
        log.info(f'Training {self._modelname} model')

        # Training termination metrics.
        MAX_EPOCHS = 999999
        best_avg_loss = float('inf')
        best_epoch = 1
        best_model_weights = copy.deepcopy(self._model.to('cpu').state_dict())

        # Early stopping metrics.
        MAX_FORGIVES = 5
        times_forgiven = 0
        
        # Learning rate scheduler and auto caster.
        lr_scheduler = ReduceLROnPlateau(self._optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        scaler = GradScaler()

        # Grayscale and Tensor transform.
        to_grayscale = Grayscale(num_output_channels=1)

        self._model.to(self._device)
        for epoch in range(1, MAX_EPOCHS):
            log.info('Epoch [{}/{}],'.format(epoch, MAX_EPOCHS))

            # Training.
            self._model.train()
            running_training_loss = 0.0
            
            for left_inputs, right_inputs, labels in training_dataloader:
                inputs = []

                for left_input, right_input in zip(left_inputs, right_inputs):
                    left_input_grayscale = to_grayscale(left_input)
                    right_input_grayscale = to_grayscale(right_input)
                    
                    input = torch.zeros((3, left_input.shape[1], left_input.shape[2]))
                    input[0] = left_input_grayscale
                    input[1] = right_input_grayscale

                    inputs.append(input)

                inputs = torch.stack(inputs)
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                
                self._optimizer.zero_grad()

                with autocast():
                    outputs = self._model(inputs)
                    current_batch_loss = self._criterion(outputs, labels)
                
                scaler.scale(current_batch_loss).backward()
                scaler.step(self._optimizer)
                scaler.update()

                # Add the loss for this batch to the running loss.
                running_training_loss += current_batch_loss.item() * inputs.size(0)
            
            # Compute the average training loss for this epoch.
            avg_training_loss = running_training_loss / len(training_dataloader.dataset)
            log.info('Training Loss: {:.4f},'.format(avg_training_loss))
            
            # Testing.
            self._model.eval()
            running_testing_loss = 0.0

            with torch.no_grad():
                for left_inputs, right_inputs, labels in testing_dataloader:
                    inputs = [] 

                    for left_input, right_input in zip(left_inputs, right_inputs):
                        left_input_grayscale = to_grayscale(left_input)
                        right_input_grayscale = to_grayscale(right_input)
                        
                        input = torch.zeros((3, left_input.shape[1], left_input.shape[2]))
                        input[0] = left_input_grayscale
                        input[1] = right_input_grayscale

                        inputs.append(input)

                    inputs = torch.stack(inputs)
                    inputs = inputs.to(self._device, non_blocking=True)
                    labels = labels.to(self._device, non_blocking=True)

                    with autocast():
                        outputs = self._model(inputs)
                        current_batch_loss = self._criterion(outputs, labels)

                    # Add the loss for this batch to the running loss.
                    running_testing_loss += current_batch_loss.item() * inputs.size(0)

            # Compute the average training loss for this epoch.
            testing_avg_loss = running_testing_loss / len(testing_dataloader.dataset)
            log.info('Testing Loss: {:.4f},'.format(testing_avg_loss))

            # Check if training should stop.
            if testing_avg_loss < best_avg_loss:
                best_avg_loss = testing_avg_loss
                best_epoch = epoch
                best_model_weights = copy.deepcopy(self._model.to('cpu').state_dict())
                self._model.to(self._device)
                times_forgiven = 0
            else:
                times_forgiven += 1
                if times_forgiven > MAX_FORGIVES:
                    log.info(f'Stopping at epoch {epoch + 1}')
                    break

            # Update the learning rate if the testing loss does not improve.
            lr_scheduler.step(testing_avg_loss)
            
        log.info(f'Using weights from epoch {best_epoch}')
        self._model.load_state_dict(best_model_weights)
        return best_epoch, best_avg_loss
    
    def evaluate_model_accuracy(self, dataloader: DataLoader) -> float:
        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} accuracy')
        
        accuracy = Accuracy(task='multiclass',
                            num_classes=self._num_output_classes)
        accuracy = accuracy.to(self._device)
        
        # Grayscale and Tensor transform.
        to_grayscale = Grayscale(num_output_channels=1)
        
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            for left_inputs, right_inputs, labels in dataloader:
                inputs = []

                for left_input, right_input in zip(left_inputs, right_inputs):
                    left_input_grayscale = to_grayscale(left_input)
                    right_input_grayscale = to_grayscale(right_input)
                    
                    input = torch.zeros((3, left_input.shape[1], left_input.shape[2]))
                    input[0] = left_input_grayscale
                    input[1] = right_input_grayscale

                    inputs.append(input)

                inputs = torch.stack(inputs)
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model(inputs)
                
                _, preds = torch.max(outputs, dim=1)
                
                accuracy(preds, labels)
        
        accuracy = accuracy.compute().cpu().item()
        log.info(f'Accuracy: {accuracy:.4f}')
        return accuracy
    
    def evaluate_model_roc(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} ROC')
        
        # Grayscale and Tensor transform.
        to_grayscale = Grayscale(num_output_channels=1)

        num_output_classes = self._num_output_classes

        probs = []
        labels = []

        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            for left_inputs, right_inputs, batch_labels in dataloader:
                inputs = []

                for left_input, right_input in zip(left_inputs, right_inputs):
                    left_input_grayscale = to_grayscale(left_input)
                    right_input_grayscale = to_grayscale(right_input)
                    
                    input = torch.zeros((3, left_input.shape[1], left_input.shape[2]))
                    input[0] = left_input_grayscale
                    input[1] = right_input_grayscale

                    inputs.append(input)

                inputs = torch.stack(inputs)
                inputs = inputs.to(self._device, non_blocking=True)
                labels.extend(batch_labels.cpu().numpy())

                with autocast():
                    outputs = self._model(inputs)

                probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        labels = label_binarize(labels, classes=np.arange(num_output_classes))

        fpr, tpr, _ = roc_curve(labels.ravel(), np.array(probs).ravel())
        
        return fpr, tpr
    
    def evaluate_model_euclidean_distances_and_save(self, dataloader: DataLoader, model_filebasename: str, filename: str):
        log = logging.getLogger(__name__)
        log.info(f'Evaluating dataset euclidean distances for {self._modelname}')

        # Grayscale and Tensor transform.
        to_grayscale = Grayscale(num_output_channels=1)

        feature_extractor = nn.Sequential(*list(self._model.children())[:-1])
        feature_extractor.to(self._device, non_blocking=True)
        
        row_labels = []
        features_list = []
        
        with torch.no_grad():
            for left_inputs, right_inputs, labels in dataloader:
                inputs = []

                for left_input, right_input in zip(left_inputs, right_inputs):
                    left_input_grayscale = to_grayscale(left_input)
                    right_input_grayscale = to_grayscale(right_input)
                    
                    input = torch.zeros((3, left_input.shape[1], left_input.shape[2]))
                    input[0] = left_input_grayscale
                    input[1] = right_input_grayscale

                    inputs.append(input)

                inputs = torch.stack(inputs)
                inputs = inputs.to(self._device, non_blocking=True)

                with autocast():
                    features = feature_extractor(inputs)
                
                features = features.view(features.size(0), -1)
                
                row_labels.extend(labels)
                features_list.append(features.cpu().numpy())

        features_list = [torch.from_numpy(features) for features in features_list]
        features = torch.cat(features_list, dim=0).half().cuda()

        row_labels = np.array(row_labels, dtype='S')
        euclidean_distance_matrix = np.empty((len(features), len(features)), dtype=np.float16)

        output_writer = OutputWriter()
        output_writer.write_hdf5_data(model_filebasename=model_filebasename, filename=filename, dataset_name='row_labels', data=row_labels)
        output_writer.write_hdf5_data(model_filebasename=model_filebasename, filename=filename, dataset_name='euclidean_distance_matrix', data=euclidean_distance_matrix)

        # The distances are computed in chunks to avoid running out of memory.
        chunk_size = 1000
        for chunk_start_idx in range(0, len(features), chunk_size):
            chunk_end_idx = chunk_start_idx + chunk_size

            features_slice = features[chunk_start_idx:chunk_end_idx]
            distances_slice = torch.empty((features_slice.shape[0], features.shape[0]), dtype=torch.float16, device='cuda')
            
            # The distances are vectorized to speed up the computation.
            for i in range(features_slice.shape[0]):
                distances_slice[i] = torch.sqrt(torch.sum((features_slice[i, None] - features) ** 2, dim=-1))
            distances_slice[distances_slice == 0] = float('nan')

            output_writer.append_hdf5_data_slice(model_filebasename=model_filebasename,
                                                 filename=filename,
                                                 dataset_name='euclidean_distance_matrix',
                                                 data_slice=distances_slice.cpu().numpy(),
                                                 start_idx=chunk_start_idx)