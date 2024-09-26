# -*- coding: utf-8 -*-
"""
Module: triplet_paired_iris_dataset.py
Author: Daniel Tebor
Description: This module contains a class for loading three images from a paired iris dataset (so actually 6 images, but concatinated to 3).
             Used for training Siamese networks with triplet loss.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from iris_dataset_processing.deterministic_paired_iris_dataset import DeterministicPairedIrisDataset
from iris_dataset_processing.iris_dataset_flags import IrisImgFlag
from iris_dataset_processing.random_sample_paired_iris_dataset import RandomSamplePairedIrisDataset


class TripletPairedIrisDataset(Dataset):
    def __init__(self, anchor_iris_dataset: DeterministicPairedIrisDataset, random_sample_iris_dataset: RandomSamplePairedIrisDataset):
        self._anchor_iris_dataset = anchor_iris_dataset
        self._random_sample_iris_dataset = random_sample_iris_dataset

    def __len__(self):
        return len(self._anchor_iris_dataset)
    
    def _img_angle_is_valid(self, anchor_img_angle: int, filename: str) -> bool:
        img_angle = filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if img_angle.startswith('N'):
            img_angle = int(img_angle[1:]) * -1
        else:
            img_angle = int(img_angle[1:])
        
        if anchor_img_angle == img_angle or anchor_img_angle - 10 == img_angle or anchor_img_angle + 10 == img_angle:
            return True
        return False

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        anchor_left_eye_image, anchor_right_eye_image, label, anchor_left_eye_filename, anchor_right_eye_filename = self._anchor_iris_dataset[index]
        
        left_eye_img_angle = anchor_left_eye_filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if left_eye_img_angle.startswith('N'):
            left_eye_img_angle = int(left_eye_img_angle[1:]) * -1
        else:
            left_eye_img_angle = int(left_eye_img_angle[1:])
        
        right_eye_img_angle = anchor_right_eye_filename.split('_')[IrisImgFlag.ANGLE_FILENAME_COMPONENT_IDX.value]
        if right_eye_img_angle.startswith('N'):
            right_eye_img_angle = int(right_eye_img_angle[1:]) * -1
        else:
            right_eye_img_angle = int(right_eye_img_angle[1:])

        positive_img_is_found = False
        while not positive_img_is_found:
            positive_left_eye_image, positive_left_eye_filename = self._random_sample_iris_dataset.get_random_left_eye_image_for_encoded_label(label.item())
            if torch.equal(positive_left_eye_image, anchor_left_eye_image):
                continue
            elif self._img_angle_is_valid(anchor_img_angle=left_eye_img_angle, filename=positive_left_eye_filename):
                continue
            positive_img_is_found = True

        positive_img_is_found = False
        while not positive_img_is_found:
            positive_right_eye_image, positive_right_eye_filename = self._random_sample_iris_dataset.get_random_right_eye_image_for_encoded_label(label.item())
            if torch.equal(positive_right_eye_image, anchor_right_eye_image):
                continue
            elif self._img_angle_is_valid(anchor_img_angle=right_eye_img_angle, filename=positive_right_eye_filename):
                continue
            positive_img_is_found = True

        negative_img_is_found = False
        while not negative_img_is_found:
            negative_left_eye_image, _, negative_left_eye_filename = self._random_sample_iris_dataset.get_random_left_eye_image_excluding_encoded_label(label.item())
            if self._img_angle_is_valid(anchor_img_angle=left_eye_img_angle, filename=negative_left_eye_filename):
                continue
            negative_img_is_found = True

        negative_img_is_found = False
        while not negative_img_is_found:
            negative_right_eye_image, _, negative_right_eye_filename = self._random_sample_iris_dataset.get_random_right_eye_image_excluding_encoded_label(label.item())
            if self._img_angle_is_valid(anchor_img_angle=right_eye_img_angle, filename=negative_right_eye_filename):
                continue
            negative_img_is_found = True

        return anchor_left_eye_image, anchor_right_eye_image, positive_left_eye_image, positive_right_eye_image, negative_left_eye_image, negative_right_eye_image, label