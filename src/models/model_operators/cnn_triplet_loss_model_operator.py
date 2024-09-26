# -*- coding: utf-8 -*-
"""
Module: cnn_siamese_model_operator.py
Author: Daniel Tebor
Description: This module contains a class for training and evaluating the ResNet50 model acheived by creating a siamese model.
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

from common.output_writer import OutputWriter
from models.model_operators.cnn_model_operator import CNNModelOperator


class CNNTripletLossModelOperator(CNNModelOperator):
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
        MAX_EPOCHS = 1000
        best_avg_loss = float('inf')
        best_epoch = 1
        best_model_weights = copy.deepcopy(self._model.to('cpu').state_dict())

        # Early stopping metrics.
        MAX_FORGIVES = 5
        times_forgiven = 0
        
        # Learning rate scheduler and auto caster.
        lr_scheduler = ReduceLROnPlateau(self._optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        scaler = GradScaler()

        self._model.to(self._device)
        for epoch in range(1, MAX_EPOCHS):
            log.info('Epoch [{}/{}],'.format(epoch, MAX_EPOCHS))

            # Training.
            self._model.train()
            running_training_loss = 0.0
            
            for anchor_inputs, positive_inputs, negative_inputs, _ in training_dataloader:
                anchor_inputs = anchor_inputs.to(self._device, non_blocking=True)
                positive_inputs = positive_inputs.to(self._device, non_blocking=True)
                negative_inputs = negative_inputs.to(self._device, non_blocking=True)
                
                self._optimizer.zero_grad()

                with autocast():
                    anchor_outputs = self._model(anchor_inputs)
                    positive_outputs = self._model(positive_inputs)
                    negative_outputs = self._model(negative_inputs)
                    current_batch_loss = self._criterion(anchor_outputs, positive_outputs, negative_outputs)
                
                scaler.scale(current_batch_loss).backward()
                scaler.step(self._optimizer)
                scaler.update()

                # Add the loss for this batch to the running loss.
                running_training_loss += current_batch_loss.item() * anchor_inputs.size(0)
            
            # Compute the average training loss for this epoch.
            avg_training_loss = running_training_loss / len(training_dataloader.dataset)
            log.info('Training Loss: {:.4f},'.format(avg_training_loss))
            
            # Testing.
            self._model.eval()
            running_testing_loss = 0.0

            with torch.no_grad():
                for anchor_inputs, positive_inputs, negative_inputs, _ in testing_dataloader:
                    anchor_inputs = anchor_inputs.to(self._device, non_blocking=True)
                    positive_inputs = positive_inputs.to(self._device, non_blocking=True)
                    negative_inputs = negative_inputs.to(self._device, non_blocking=True)

                    with autocast():
                        anchor_outputs = self._model(anchor_inputs)
                        positive_outputs = self._model(positive_inputs)
                        negative_outputs = self._model(negative_inputs)
                        current_batch_loss = self._criterion(anchor_outputs, positive_outputs, negative_outputs)

                    # Add the loss for this batch to the running loss.
                    running_testing_loss += current_batch_loss.item() * anchor_inputs.size(0)

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

    def evaluate_model_accuracy(self, anchor_dataloader: DataLoader, validation_dataloader: DataLoader) -> float:
        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} accuracy')
        
        accuracy = Accuracy(task='multiclass',
                            num_classes=self._num_output_classes)
        accuracy = accuracy.to(self._device)
        
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            anchor_outputs = []
            anchor_labels = []

            for anchor_inputs, labels, _ in anchor_dataloader:
                anchor_inputs = anchor_inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model.forward(anchor_inputs)
                    anchor_outputs.extend(outputs.cpu().numpy())

                anchor_labels.extend(labels)

            for validation_inputs, labels, _ in validation_dataloader:
                validation_inputs = validation_inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model.forward(validation_inputs)
                    
                outputs = outputs.cpu().numpy()

                predicted_labels = []

                for validation_output in outputs:
                    distances = [np.linalg.norm(validation_output - anchor_output) for anchor_output in anchor_outputs]
                    min_distance_index = np.argmin(distances)

                    predicted_labels.append(anchor_labels[min_distance_index])

                predicted_labels = torch.tensor(predicted_labels, dtype=torch.long)
                accuracy.update(predicted_labels, labels)

        accuracy = accuracy.compute().cpu().item()
        log.info(f'Accuracy: {accuracy:.4f}')
        return accuracy
    
    def evaluate_model_accuracy_roc(self, anchor_dataloader: DataLoader, validation_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} ROC')

        probs = []
        labels = []
        classes = []

        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            anchor_outputs = []
            anchor_labels = []

            for anchor_inputs, batch_labels, _ in anchor_dataloader:
                anchor_inputs = anchor_inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model.forward(anchor_inputs)
                    anchor_outputs.extend(outputs.cpu().numpy())

                anchor_labels.extend(batch_labels)

            classes = [label.item() for label in anchor_labels]

            for validation_inputs, batch_labels, _ in validation_dataloader:
                validation_inputs = validation_inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model.forward(validation_inputs)
                
                outputs = outputs.cpu().numpy()
                
                for validation_output in outputs:
                    distances = [np.linalg.norm(validation_output - anchor_output) for anchor_output in anchor_outputs]
                    
                    # Normalization.
                    distances = np.array(distances)
                    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

                    # Invert the distances to effectively get prediction probabilities.
                    distances = 1 - distances

                    probs.extend(distances)
                
                    # Get the predicted label.
                    max_distance_index = np.argmax(distances)
                    predicted_label = anchor_labels[max_distance_index]

                    labels.append(predicted_label)

        labels = label_binarize(labels, classes=np.array(classes))

        fpr, tpr, _ = roc_curve(labels.ravel(), np.array(probs).ravel())
        
        return fpr, tpr
    
    def evaluate_model_euclidean_distances_and_save(self, dataloader: DataLoader, model_filebasename: str, filename: str):
        log = logging.getLogger(__name__)
        log.info(f'Evaluating dataset euclidean distances for {self._modelname}')

        self._model.to(self._device, non_blocking=True)
        
        row_labels = []
        features_list = []
        
        with torch.no_grad():
            for inputs, _, img_filenames in dataloader:
                inputs = inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    features = self._model(inputs)
                
                features = features.view(features.size(0), -1)
                
                row_labels.extend(img_filenames)
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