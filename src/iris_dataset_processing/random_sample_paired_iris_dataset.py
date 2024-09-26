# -*- coding: utf-8 -*-
"""
Module: random_sampled_paired_iris_dataset.py
Author: Daniel Tebor
Description: This module contains a class for loading the paired iris dataset. 
             Mainly used for feeding left and right iris images that are randomly sampled into a model simultaneously.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from iris_dataset_processing.iris_dataset import IrisDataset


class RandomSamplePairedIrisDataset(Dataset):
    def __init__(self, left_iris_dataset: IrisDataset, right_iris_dataset: IrisDataset, times_to_repeat_label: int = 1):
        self._left_iris_dataset = left_iris_dataset
        self._right_iris_dataset = right_iris_dataset
        self._labels = np.repeat(np.intersect1d(left_iris_dataset.labels_encoded, right_iris_dataset.labels_encoded), times_to_repeat_label)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, str, str]:
        label = self._labels[index]
        left_eye_image, left_eye_img_filename = self._left_iris_dataset.get_random_img_for_encoded_label(label)
        right_eye_image, right_eye_img_filename = self._right_iris_dataset.get_random_img_for_encoded_label(label)
        return left_eye_image, right_eye_image, torch.tensor(label), left_eye_img_filename, right_eye_img_filename

    def __len__(self):
        return len(self._labels)
    
    def get_random_imgs_for_encoded_label(self, label: int) -> Tuple[np.ndarray, np.ndarray]:
        left_eye_image = self._left_iris_dataset.get_random_img_for_encoded_label(label)
        right_eye_image = self._right_iris_dataset.get_random_img_for_encoded_label(label)
        return left_eye_image, right_eye_image
    
    def get_random_left_eye_image_for_encoded_label(self, label: int) -> Tuple[np.ndarray, str]:
        return self._left_iris_dataset.get_random_img_for_encoded_label(label)
    
    def get_random_right_eye_image_for_encoded_label(self, label: int) -> Tuple[np.ndarray, str]:
        return self._right_iris_dataset.get_random_img_for_encoded_label(label)
    
    def get_random_imgs_excluding_encoded_label(self, label: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        left_eye_image, img_label = self._left_iris_dataset.get_random_img_excluding_encoded_label(label)
        right_eye_image = self._right_iris_dataset.get_random_img_for_encoded_label(img_label.item())
        return left_eye_image, right_eye_image, img_label
    
    def get_random_left_eye_image_excluding_encoded_label(self, label: int) -> Tuple[np.ndarray, str]:
        return self._left_iris_dataset.get_random_img_excluding_encoded_label(label)
    
    def get_random_right_eye_image_excluding_encoded_label(self, label: int) -> Tuple[np.ndarray, str]:
        return self._right_iris_dataset.get_random_img_excluding_encoded_label(label)

    @property
    def left_iris_dataset(self) -> IrisDataset:
        return self._left_iris_dataset

    @property
    def right_iris_dataset(self) -> IrisDataset:
        return self._right_iris_dataset

