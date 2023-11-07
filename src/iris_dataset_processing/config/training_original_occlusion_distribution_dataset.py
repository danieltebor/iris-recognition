# -*- coding: utf-8 -*-
"""
Module: training_original_occlusion_distribution_dataset.py
Author: Daniel Tebor
Description: This module contains the configuration for training a model on the original iris occlusion distribution.
"""

from iris_dataset_processing.iris_dataset_flags import *


TRAINING_DATASET_PATH = '/mnt/d/Iris Recognition Research/OcularIris Static Cropped Images'
TRAINING_DERIVED_MODEL_DESCRIPTOR = 'original_occlusion_distribution'

TRAINING_COHORTS = [IrisDatasetCohort.ORIGINAL_OCCLUSION_DISTRIBUTION]
TRAINING_IMG_EXCLUSION_FLAGS = []
TRAINING_BATCH_SIZE = 128