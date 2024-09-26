# -*- coding: utf-8 -*-
"""
Module: triplet_paired_iris_dataset.py
Author: Daniel Tebor
Description: This module contains a class for loading three images from an iris dataset. Used for training Siamese networks with triplet loss.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from iris_dataset_processing.iris_dataset import IrisDataset
from iris_dataset_processing.iris_dataset_flags import IrisImgFlag


class TripletIrisDataset(Dataset):
    def __init__(self, anchor_iris_dataset: IrisDataset, sample_iris_dataset: IrisDataset):
        self._anchor_iris_dataset = anchor_iris_dataset
        self._sample_iris_dataset = sample_iris_dataset

    def __len__(self):
        return len(self._sample_iris_dataset)
    
    def _img_angle_is_valid(self, positive_img_angle: int, filename: str) -> bool:
        img_angle = filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if img_angle.startswith('N'):
            img_angle = int(img_angle[1:]) * -1
        else:
            img_angle = int(img_angle[1:])
        
        if positive_img_angle == img_angle or positive_img_angle - 10 == img_angle or positive_img_angle + 10 == img_angle:
            return True
        return False

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        positive_eye_img, label, positive_filename = self._sample_iris_dataset[idx]
        img_angle = positive_filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if img_angle.startswith('N'):
            img_angle = int(img_angle[1:]) * -1
        else:
            img_angle = int(img_angle[1:])

        anchor_img_is_found = False
        while not anchor_img_is_found:
            anchor_eye_img, anchor_filename = self._anchor_iris_dataset.get_random_img_for_encoded_label(label.item())
            if torch.equal(anchor_eye_img, positive_eye_img):
                continue
            if self._img_angle_is_valid(positive_img_angle=img_angle, filename=anchor_filename):
                continue
            anchor_img_is_found = True

        """
        anchor_eye_img, label, anchor_filename = self._anchor_iris_dataset[idx]
        img_angle = anchor_filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if img_angle.startswith('N'):
            img_angle = int(img_angle[1:]) * -1
        else:
            img_angle = int(img_angle[1:])

        positive_img_is_found = False
        while not positive_img_is_found:
            positive_eye_img, positive_filename = self._sample_iris_dataset.get_random_img_for_encoded_label(label.item())
            if torch.equal(anchor_eye_img, positive_eye_img):
                continue
            if self._img_angle_is_valid(anchor_img_angle=img_angle, filename=positive_filename):
                continue
            positive_img_is_found = True
        """

        negative_img_is_found = False
        while not negative_img_is_found:
            negative_eye_img, _, negative_filename = self._sample_iris_dataset.get_random_img_excluding_encoded_label(label.item())
            if self._img_angle_is_valid(positive_img_angle=img_angle, filename=negative_filename): 
                continue
            negative_img_is_found = True
        
        return anchor_eye_img, positive_eye_img, negative_eye_img, label