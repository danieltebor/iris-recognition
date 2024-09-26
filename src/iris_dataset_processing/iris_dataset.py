# -*- coding: utf-8 -*-
"""
Module: iris_dataset.py
Author: Daniel Tebor
Description: This module contains a class for loading the iris dataset.
"""

import itertools
import logging
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from iris_dataset_processing.iris_dataset_flags import *
from iris_dataset_processing.circle_crop import CircleCrop


class IrisDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 cohorts: list[IrisDatasetCohort],
                 img_exclusion_flags: list[IrisImgFlag],
                 img_input_shape: Tuple[int, int, int],
                 label_encoder: LabelEncoder = None,
                 should_use_augmentation: bool = False):
        self._data_path = data_path

        # Build list of image paths.
        img_paths = []
        img_filenames = []
        for cohort in cohorts:
            cohort_path = os.path.join(data_path, cohort.value)
            with os.scandir(cohort_path) as entries:
                for entry in entries:
                    if entry.is_file() and not self._img_file_contains_flag(entry.name, img_exclusion_flags):
                        img_paths.append(entry.path)
                        img_filenames.append(entry.name)

        self._img_paths = np.array(img_paths)
        self._img_filenames = np.array(img_filenames)

        # Build list of image labels.
        labels = []
        for img_filename in self._img_filenames:
            label = img_filename.split('_')[IrisImgFlag.SUBJECT_FILENAME_COMPONENT_IDX.value][1:]
            labels.append(int(label))

        # Encode the labels to consecutive integers.
        self._label_encoder = label_encoder
        if self._label_encoder is None:
            self._label_encoder = LabelEncoder()
            self._labels = self._label_encoder.fit_transform(labels)
        else:
            self._labels = self._label_encoder.transform(labels)

        self._labels = np.array(self._labels)

        # Build image transform.
        transform_composition = [
            transforms.Resize(img_input_shape[1:3]),
            transforms.ToTensor(),
            # Normalize to ImageNet standard.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ]

        # Add augmentation transforms if enabled.
        if should_use_augmentation:
            transform_composition.insert(0, transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)))
            affine_transform = transforms.RandomAffine(degrees=20,
                                                       translate=(0.1, 0.1),
                                                       interpolation=InterpolationMode.BICUBIC)
            transform_composition.insert(0, affine_transform)
            #transform_composition.insert(0, CircleCrop(radius_percent_range=(0.25, 0.5), mode='inner'))
            #transform_composition.insert(0, CircleCrop(radius_percent_range=(0.8, 1.0), mode='outer'))
            transform_composition.insert(0, transforms.RandomResizedCrop(size=img_input_shape[1:3], scale=(0.95, 1.0), ratio=(1, 1)))

        self._transform = transforms.Compose(transform_composition)

        log = logging.getLogger(__name__)
        log.info(f'Configured dataset of {len(self._img_paths)} images and {self.num_classes} classes from "{data_path}"')

    def __len__(self) -> int:
        return len(self._img_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self._img_paths[idx]
        
        img = Image.open(img_path)
        if len(img.getbands()) == 1:
            img = img.convert('RGB')
        img = self._transform(img)

        label = torch.tensor(self._labels[idx])
        
        img_filename = self._img_filenames[idx]
        
        return img, label, img_filename
    
    def __str__(self) -> str:
        img_file_str = '['
        for img_file in self._img_paths:
            img_dir = os.path.dirname(img_file)
            img_filename = os.path.basename(img_file)
            img_file_str += '{0}/{1}, '.format(img_dir, img_filename)
        img_file_str = img_file_str[:-2]
        img_file_str += ']'
        
        return img_file_str
    
    def _img_file_contains_flag(self, filename: str, flags: list[IrisImgFlag]) -> bool:
        # Flatten the flags list.
        flags = list(itertools.chain(*[flag.value if isinstance(flag.value, list) else [flag.value] for flag in flags]))
        
        filename_components = filename.split('.')[0].split('_')
        for flag in flags:
            component_idx = 0
            flag_found = False

            for component in filename_components:
                # Check if the component index is correct if the flag is an img num flag.
                if flag in IrisImgFlag.ALL_IMG_NUM_FLAGS.value and not component_idx == IrisImgFlag.IMG_NUM_FILENAME_COMPONENT_IDX.value:
                    continue

                if flag in component:
                    flag_found = True
                    break

                component_idx += 1

            if flag_found:
                return True
        
        return False
    
    def get_random_img_for_encoded_label(self, label: int) -> Tuple[torch.Tensor, str]:
        label_indices = [i for i, l in enumerate(self.labels_encoded) if l == label]
        random_index = random.choice(label_indices)
        return self.__getitem__(random_index)[0], self.__getitem__(random_index)[2]
    
    def get_random_img_excluding_encoded_label(self, label: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        label_indices = [i for i, l in enumerate(self.labels_encoded) if l != label]
        random_index = random.choice(label_indices)
        img, label, filename = self.__getitem__(random_index)
        return img, label, filename

    @property
    def img_paths(self) -> np.ndarray[str]:
        return self._img_paths
    
    @property
    def img_filenames(self) -> np.ndarray[str]:
        return self._img_filenames
    
    @property
    def img_tensors(self) -> np.ndarray[torch.Tensor]:
        img_tensors = np.array([])
        img_tensors.resize(len(self._img_filenames))
        for i in range(len(self._img_filenames)):
            img_tensor, _, _ = self.__getitem__(i)
            img_tensors.append(img_tensor)
        return img_tensors

    @property
    def labels(self) -> np.ndarray[int]:
        return set(self.label_encoder.inverse_transform(self._labels))
    
    @property
    def labels_encoded(self) -> np.ndarray[int]:
        return self._labels
    
    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder
    
    @property
    def num_classes(self) -> int:
        return len(set(self.labels_encoded))
    
    @property
    def transform(self) -> transforms.Compose:
        return self._transform