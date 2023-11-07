# -*- coding: utf-8 -*-
"""
Module: testing_0_to_90_percent_occlusion_dataset.py
Author: Daniel Tebor
Description: This module contains the configuration for testing a model on a dataset of 0 to 90 percent occlused images.
"""

from iris_dataset_processing.iris_dataset_flags import *


TESTING_DATASET_PATH = '/mnt/d/Iris Recognition Research/OcularIris Left Cropped Images'

TESTING_COHORTS = [
    IrisDatasetCohort.PERCENT_OCCLUSION_00,
    IrisDatasetCohort.PERCENT_OCCLUSION_10,
    IrisDatasetCohort.PERCENT_OCCLUSION_20,
    IrisDatasetCohort.PERCENT_OCCLUSION_30,
    IrisDatasetCohort.PERCENT_OCCLUSION_40,
    IrisDatasetCohort.PERCENT_OCCLUSION_50,
    IrisDatasetCohort.PERCENT_OCCLUSION_60,
    IrisDatasetCohort.PERCENT_OCCLUSION_70,
    IrisDatasetCohort.PERCENT_OCCLUSION_80,
    IrisDatasetCohort.PERCENT_OCCLUSION_90,
]
TESTING_IMG_EXCLUSION_FLAGS = [IrisImgFlag.OFF_ANGLE_FLAGS]
TESTING_BATCH_SIZE = 128