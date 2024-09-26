# -*- coding: utf-8 -*-
"""
Module: subject_complete_occlusion_dataset.py
Author: Daniel Tebor
Description: This module contains the configuration for loading a dataset with every subject present.
"""

from iris_dataset_processing.iris_dataset_flags import *


SUBJECT_COMPLETE_OCCLUSION_DATASET_PATH = '/mnt/d/Iris Recognition Research/OcularIris Static Cropped Images'

SUBJECT_COMPLETE_COHORTS = [IrisDatasetCohort.ORIGINAL_OCCLUSION_DISTRIBUTION]
SUBJECT_COMPLETE_IMG_EXCLUSION_FLAGS = []