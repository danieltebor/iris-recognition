# -*- coding: utf-8 -*-
"""
Module: deterministic_paired_iris_dataset.py
Author: Daniel Tebor
Description: This module contains a class for loading the paired iris dataset. 
             Mainly used for feeding left and right iris images into a model simultaneously.
"""

from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from iris_dataset_processing.iris_dataset import IrisDataset
from iris_dataset_processing.iris_dataset_flags import *


class DeterministicPairedIrisDataset(Dataset):
    def __init__(self, left_iris_dataset: IrisDataset, right_iris_dataset: IrisDataset):
        left_eye_img_paths = left_iris_dataset.img_paths.copy()
        right_eye_img_paths = right_iris_dataset.img_paths.copy()

        left_eye_img_filenames = left_iris_dataset.img_filenames.copy()
        right_eye_img_filenames = right_iris_dataset.img_filenames.copy()
        left_eye_img_filenames = np.char.replace(left_eye_img_filenames, '_L_', '_')
        right_eye_img_filenames = np.char.replace(right_eye_img_filenames, '_R_', '_')

        left_eye_img_paths_dict = dict(np.column_stack((left_eye_img_filenames, left_eye_img_paths)))
        right_eye_img_paths_dict = dict(np.column_stack((right_eye_img_filenames, right_eye_img_paths)))

        left_and_right_eye_filename_intersection = np.intersect1d(left_eye_img_filenames, right_eye_img_filenames)

        left_eye_img_paths_dict = {k: v for k, v in left_eye_img_paths_dict.items() if k in left_and_right_eye_filename_intersection}
        right_eye_img_paths_dict = {k: v for k, v in right_eye_img_paths_dict.items() if k in left_and_right_eye_filename_intersection}

        left_eye_img_filenames, self._left_eye_img_paths = map(np.array, zip(*left_eye_img_paths_dict.items()))
        right_eye_img_filenames, self._right_eye_img_paths = map(np.array, zip(*right_eye_img_paths_dict.items()))

        self._left_eye_img_filenames = np.array([filename[:5] + 'L_' + filename[5:] for filename in left_eye_img_filenames])
        self._right_eye_img_filenames = np.array([filename[:5] + 'R_' + filename[5:] for filename in right_eye_img_filenames])

        labels = []
        for img_filename in self._left_eye_img_filenames:
            label = img_filename.split('_')[IrisImgFlag.SUBJECT_FILENAME_COMPONENT_IDX.value][1:]
            labels.append(int(label))

        self._label_encoder = left_iris_dataset.label_encoder
        self._labels = np.array(self._label_encoder.transform(labels))

        self._transform = left_iris_dataset.transform

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, str, str]:
        left_eye_image = self._left_eye_img_paths[index]
        left_eye_image = Image.open(left_eye_image)
        if len(left_eye_image.getbands()) == 1:
            left_eye_image = left_eye_image.convert('RGB')
        left_eye_image = self._transform(left_eye_image)

        right_eye_image = self._right_eye_img_paths[index]
        right_eye_image = Image.open(right_eye_image)
        if len(right_eye_image.getbands()) == 1:
            right_eye_image = right_eye_image.convert('RGB')
        right_eye_image = self._transform(right_eye_image)

        left_eye_image_filename = self._left_eye_img_filenames[index]
        right_eye_image_filename = self._right_eye_img_filenames[index]

        label = self._labels[index]
        return left_eye_image, right_eye_image, torch.tensor(label), left_eye_image_filename, right_eye_image_filename

    def __len__(self):
        return len(self._left_eye_img_paths)
    
    @property
    def labels_encoded(self) -> np.ndarray[int]:
        return self._labels

    @property
    def left_eye_img_paths(self) -> np.ndarray[str]:
        return self._left_eye_img_paths
    
    @property
    def right_eye_img_paths(self) -> np.ndarray[str]:
        return self._right_eye_img_paths