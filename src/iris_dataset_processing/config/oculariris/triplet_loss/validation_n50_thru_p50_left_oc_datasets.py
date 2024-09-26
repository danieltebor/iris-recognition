# -*- coding: utf-8 -*-
from iris_dataset_processing.iris_dataset_flags import *


VALIDATION_DATASET_PATH = './data'

VALIDATION_CONFIGS = {}

# -50
_negative_50_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['-50'] = _negative_50_config

# -40
_negative_40_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['-40'] = _negative_40_config

# -30
_negative_30_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['-30'] = _negative_30_config

# -20
_negative_20_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['-20'] = _negative_20_config

# -10
_negative_10_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['-10'] = _negative_10_config

# 0
_zero_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['0'] = _zero_config

# 10
_positive_10_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['10'] = _positive_10_config

# 20
_positive_20_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['20'] = _positive_20_config

# 30
_positive_30_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P40,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['30'] = _positive_30_config

# 40
_positive_40_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P50],
    'batch_size': 256
}
VALIDATION_CONFIGS['40'] = _positive_40_config

# 50
_positive_50_config = {
    'cohorts': [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TESTING_OC_LEFT],
    'img_exclusion_flags': [IrisImgFlag.ANGLE_N50,
                            IrisImgFlag.ANGLE_N40,
                            IrisImgFlag.ANGLE_N30,
                            IrisImgFlag.ANGLE_N20,
                            IrisImgFlag.ANGLE_N10,
                            IrisImgFlag.ANGLE_P00,
                            IrisImgFlag.ANGLE_P10,
                            IrisImgFlag.ANGLE_P20,
                            IrisImgFlag.ANGLE_P30,
                            IrisImgFlag.ANGLE_P40],
    'batch_size': 256
}
VALIDATION_CONFIGS['50'] = _positive_50_config