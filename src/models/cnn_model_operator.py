# -*- coding: utf-8 -*-
"""
Module: cnn_model_trainer.py
Author: Daniel Tebor
Description: This module contains a class for training and evaluating a CNN model.
"""

import copy
import logging

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


class CNNModelOperator:
    """
    A class for training, evaluating, and saving a convolutional neural network model.
    
    Attributes:
        _model (nn.Module): The PyTorch model to train and evaluate.
        _modelname (str): The name of the model.
        _num_output_classes (int): The number of output classes for the model.
        _criterion (nn.Module): The loss function to use for training.
        _optimizer (optim.Optimizer): The optimizer to use for training.
        _device (torch.device): The device to use for training and evaluation.
        
    Methods:
        train(training_dataloader: DataLoader, testing_dataloader: DataLoader) -> int:
            Trains the model using the provided training dataset and dataloader. Training stops once the performance on the 
            testing dataset stops improving.
            
        evaluate_model_accuracy(dataloader: DataLoader) -> float:
            Evaluates the accuracy of the model on the given dataloader.
            
        evaluate_model_roc(dataloader: DataLoader) -> (np.ndarray, np.ndarray):
            Evaluates the receiver operating characteristic (ROC) of the trained model on the given data.
            
        evaluate_model_euclidean_distances_and_save(dataloader: DataLoader, filename: str):
            Evaluates the euclidean distances of the dataset and saves it to a file.
    """
    
    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 num_output_classes: int,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer):
        self._model = model
        self._modelname = model_name
        self._num_output_classes = num_output_classes
        self._criterion = criterion
        self._optimizer = optimizer
        
        self._device = torch.device('cuda:0')
        self._model.to(self._device)
        
    def train(self, training_dataloader: DataLoader, testing_dataloader: DataLoader) -> (int, int):
        """
        Trains the model using the provided training dataset and dataloader. Training stops once the performance on the 
        testing dataset stops improving.

        Args:
            training_dataloader (DataLoader): The dataloader for the training dataset.
            testing_dataloader (DataLoader): The dataloader for the testing dataset.

        Returns:
            int: The epoch number with the best validation loss.
        """
        
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
            
            for inputs, labels, _ in training_dataloader:
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
                for inputs, labels, _ in testing_dataloader:
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
        """
        Evaluates the accuracy of the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to evaluate the model on.

        Returns:
            float: The accuracy of the model on the given dataloader.
        """
        
        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} accuracy')
        
        accuracy = Accuracy(task='multiclass',
                            num_classes=self._num_output_classes)
        accuracy = accuracy.to(self._device)
        
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                
                with autocast():
                    outputs = self._model(inputs)
                
                _, preds = torch.max(outputs, dim=1)
                
                accuracy(preds, labels)
        
        accuracy = accuracy.compute().cpu().item()
        log.info(f'Accuracy: {accuracy:.4f}')
        return accuracy

    def evaluate_model_roc(self, dataloader: DataLoader) -> (np.ndarray, np.ndarray):
        """
        Evaluates the receiver operating characteristic (ROC) of the trained model on the given data.

        Args:
            dataloader (DataLoader): The data to evaluate the model on.

        Returns:
            Tuple of two numpy arrays: The false positive rates (fpr) and true positive rates (tpr).
        """

        log = logging.getLogger(__name__)
        log.info(f'Computing {self._modelname} ROC')
        
        num_output_classes = self._num_output_classes

        probs = []
        labels = []

        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            for inputs, batch_labels, _ in dataloader:
                inputs = inputs.to(self._device, non_blocking=True)
                labels.extend(batch_labels.cpu().numpy())

                with autocast():
                    outputs = self._model(inputs)

                probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        labels = label_binarize(labels, classes=np.arange(num_output_classes))

        fpr, tpr, _ = roc_curve(labels.ravel(), np.array(probs).ravel())
        
        return fpr, tpr

    def evaluate_model_euclidean_distances_and_save(self, dataloader: DataLoader, model_filebasename: str, filename: str):
        """
        Evaluates the euclidean distances of the dataset and saves it to a file.

        Args:
            dataloader: A PyTorch DataLoader object containing the dataset to evaluate.
            savefile: A string representing the path to the file where the euclidean distances will be saved.

        Returns:
            None
        """

        log = logging.getLogger(__name__)
        log.info(f'Evaluating dataset euclidean distances for {self._modelname}')
        
        feature_extractor = torch.nn.Sequential(*list(self._model.children())[:-1])
        feature_extractor.to(self._device, non_blocking=True)
        
        log.info('Collecting features')
        
        row_labels = []
        features_list = []

        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            for inputs, _, img_filenames in dataloader:
                inputs = inputs.to(self._device, non_blocking=True)
                
                with autocast():
                    features = feature_extractor(inputs)
                
                features = features.view(features.size(0), -1)
                
                row_labels.extend(img_filenames)
                features_list.append(features.cpu().numpy())

        log.info('Computing euclidean distances')

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
            log.info(f'Computing slice from {chunk_start_idx} to {chunk_end_idx}')

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