# -*- coding: utf-8 -*-
from iris_dataset_processing.iris_dataset_flags import *


TRAINING_DATASET_PATH = './data/'

TRAINING_COHORTS = [IrisDatasetCohort.OCULARIRIS_TRIPLET_LOSS_TRAINING_OC_LEFT]
TRAINING_IMG_EXCLUSION_FLAGS = []
TRAINING_BATCH_SIZE = 24