# -*- coding: utf-8 -*-
"""
Module: iris_dataset_flags.py
Author: Daniel Tebor
Description: This module contains enums for the iris dataset.
"""

from enum import Enum


class IrisDatasetCohort(Enum):
    """
    Enum of the different cohorts of iris images in the dataset.
    """

    NORMALIZED = 'Normalized'
    OCULAR = 'Ocular'
    OCULAR_IRIS = 'OcularIris'
    OCULAR_NO_IRIS = 'OcularNoIris'
    PERIOCULAR = 'Periocular'
    PERIOCULAR_IRIS = 'PeriocularIris'
    PERIOCULAR_NO_IRIS = 'PeriocularNoIris'
    
    ORIGINAL_OCCLUSION_DISTRIBUTION = 'original_occlusion_distribution'
    PERCENT_OCCLUSION_00 = '0_percent_occlusion'
    PERCENT_OCCLUSION_10 = '10_percent_occlusion'
    PERCENT_OCCLUSION_20 = '20_percent_occlusion'
    PERCENT_OCCLUSION_30 = '30_percent_occlusion'
    PERCENT_OCCLUSION_40 = '40_percent_occlusion'
    PERCENT_OCCLUSION_50 = '50_percent_occlusion'
    PERCENT_OCCLUSION_60 = '60_percent_occlusion'
    PERCENT_OCCLUSION_70 = '70_percent_occlusion'
    PERCENT_OCCLUSION_80 = '80_percent_occlusion'
    PERCENT_OCCLUSION_90 = '90_percent_occlusion'

from enum import Enum

class IrisImgFlag(Enum):
    """
    Enum for iris image flags. The values are components of file names in the iris dataset.
    """

    SUBJECT = 's'
    L_EYE = 'L'
    R_EYE = 'R'
    FRONTAL_IMG = 'Frontal'
    OFF_ANGLE_IMG = 'OffAngle'
    FRONTAL_CAM = 'Fc'
    ORBITAL_CAM = 'Oc'

    SUBJECT_FILENAME_COMPONENT_IDX = 0
    EYE_FILENAME_COMPONENT_IDX = 1
    IMG_IS_ANGLED_COMPONENT = 2
    CAMERA_FILENAME_COMPONENT_IDX = 3

    # 11 angles are taken per eye, resulting in 11 angle flags.
    ANGLE_P00 = 'P00'
    ANGLE_N10 = 'N10'
    ANGLE_P10 = 'P10'
    ANGLE_N20 = 'N20'
    ANGLE_P20 = 'P20'
    ANGLE_N30 = 'N30'
    ANGLE_P30 = 'P30'
    ANGLE_N40 = 'N40'
    ANGLE_P40 = 'P40'
    ANGLE_N50 = 'N50'
    ANGLE_P50 = 'P50'

    NEG_ANGLE_FLAGS = [ANGLE_N10, ANGLE_N20, ANGLE_N30, ANGLE_N40, ANGLE_N50]
    POS_ANGLE_FLAGS = [ANGLE_P10, ANGLE_P20, ANGLE_P30, ANGLE_P40, ANGLE_P50]
    OFF_ANGLE_FLAGS = NEG_ANGLE_FLAGS + POS_ANGLE_FLAGS
    ALL_ANGLE_FLAGS = [ANGLE_P00] + OFF_ANGLE_FLAGS

    ANGLE_FILENAME_COMPONENT_IDX = 4
    
    # 10 images are taken per angle, resulting in 10 files per angle marked 1 through 10.
    IMG_NUM_001 = '001'
    IMG_NUM_002 = '002'
    IMG_NUM_003 = '003'
    IMG_NUM_004 = '004'
    IMG_NUM_005 = '005'
    IMG_NUM_006 = '006'
    IMG_NUM_007 = '007'
    IMG_NUM_008 = '008'
    IMG_NUM_009 = '009'
    IMG_NUM_010 = '010'

    ALL_IMG_NUM_FLAGS = [
        IMG_NUM_001,
        IMG_NUM_002,
        IMG_NUM_003,
        IMG_NUM_004,
        IMG_NUM_005,
        IMG_NUM_006,
        IMG_NUM_007,
        IMG_NUM_008,
        IMG_NUM_009,
        IMG_NUM_010
    ]

    IMG_NUM_FILENAME_COMPONENT_IDX = 5