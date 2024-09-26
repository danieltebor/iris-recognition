# -*- coding: utf-8 -*-
from iris_dataset_processing.iris_dataset_flags import *


TRAINING_DATASET_PATH = './data/'

TRAINING_COHORTS = [IrisDatasetCohort.OCULARIRIS_FC_LEFT, IrisDatasetCohort.OCULARIRIS_FC_RIGHT]
LEFT_EYE_TRAINING_COHORTS = [IrisDatasetCohort.OCULARIRIS_FC_LEFT]
RIGHT_EYE_TRAINING_COHORTS = [IrisDatasetCohort.OCULARIRIS_FC_RIGHT]
TRAINING_IMG_EXCLUSION_FLAGS = []
TRAINING_BATCH_SIZE = 96