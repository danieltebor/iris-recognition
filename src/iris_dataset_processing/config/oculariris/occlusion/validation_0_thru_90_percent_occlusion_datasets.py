# -*- coding: utf-8 -*-
"""
Module: validation_0_to_90_percent_occlusion.py
Author: Daniel Tebor
Description: This module contains the configurations for validating the trained model at each occlusion percentage.
"""

from iris_dataset_processing.iris_dataset_flags import *


VALIDATION_DATASET_PATH = '/mnt/d/Iris Recognition Research/OcularIris Left Cropped Images'

VALIDATION_CONFIGS = {}

# 0% occlusion.
_0_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_00],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['0_percent_occlusion_dataset'] = _0_percent_occlusion_config

# 10% occlusion.
_10_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_10],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['10_percent_occlusion_dataset'] = _10_percent_occlusion_config

# 20% occlusion.
_20_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_20],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['20_percent_occlusion_dataset'] = _20_percent_occlusion_config

# 30% occlusion.
_30_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_30],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['30_percent_occlusion_dataset'] = _30_percent_occlusion_config

# 40% occlusion.
_40_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_40],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['40_percent_occlusion_dataset'] = _40_percent_occlusion_config

# 50% occlusion.
_50_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_50],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['50_percent_occlusion_dataset'] = _50_percent_occlusion_config

# 60% occlusion.
_60_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_60],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['60_percent_occlusion_dataset'] = _60_percent_occlusion_config

# 70% occlusion.
_70_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_70],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['70_percent_occlusion_dataset'] = _70_percent_occlusion_config

# 80% occlusion.
_80_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_80],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['80_percent_occlusion_dataset'] = _80_percent_occlusion_config

# 90% occlusion.
_90_percent_occlusion_config = {
    'cohorts': [IrisDatasetCohort.PERCENT_OCCLUSION_90],
    'img_exclusion_flags': [IrisImgFlag.OFF_ANGLE_FLAGS],
    'batch_size': 128,
}
VALIDATION_CONFIGS['90_percent_occlusion_dataset'] = _90_percent_occlusion_config