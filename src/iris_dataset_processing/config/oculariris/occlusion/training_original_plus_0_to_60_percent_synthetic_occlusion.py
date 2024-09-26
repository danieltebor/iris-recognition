# -*- coding: utf-8 -*-
"""
Module: training_original_plus_0_to_60_percent_synthetic_occlusion.py
Author: Daniel Tebor
Description: This module contains the configuration for training a model 
             on the original iris occlusion distribution plus 0 to 60 percent synthetic occlusion.
"""

from iris_dataset_processing.iris_dataset_flags import *


TRAINING_DATASET_PATH = '/mnt/d/Iris Recognition Research/OcularIris Static Cropped Images'
TRAINING_DERIVED_MODEL_DESCRIPTOR = 'original_plus_0_to_60_percent_synthetic_occlusion'

TRAINING_COHORTS = [
    IrisDatasetCohort.ORIGINAL_OCCLUSION_DISTRIBUTION,
    IrisDatasetCohort.PERCENT_OCCLUSION_00,
    IrisDatasetCohort.PERCENT_OCCLUSION_10,
    IrisDatasetCohort.PERCENT_OCCLUSION_20,
    IrisDatasetCohort.PERCENT_OCCLUSION_30,
    IrisDatasetCohort.PERCENT_OCCLUSION_40,
    IrisDatasetCohort.PERCENT_OCCLUSION_50,
    IrisDatasetCohort.PERCENT_OCCLUSION_60,
]
TRAINING_IMG_EXCLUSION_FLAGS = []
TRAINING_BATCH_SIZE = 128