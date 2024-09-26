# -*- coding: utf-8 -*-
import logging

import torch

#from scripts.left_and_right_eye_combining.resnet50_left_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.resnet50_right_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.resnet50_left_and_right_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.resnet50_merged_rgb_paired_left_and_right_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.resnet50_wide_input_paired_left_and_right_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.triplet_loss.left_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.triplet_loss.right_oculariris import train, evaluate_model
from scripts.left_and_right_eye_combining.triplet_loss.left_and_right_oculariris import train, evaluate_model
#from scripts.left_and_right_eye_combining.resnet50_siamese_wide_input_paired_left_and_right_oculariris import train, evaluate_model


SHOULD_TRAIN = True
SHOULD_EVALUATE = True
SHOULD_EVALUATE_ACCURACY = True
SHOULD_EVALUATE_ACCURACY_ROC = True
SHOULD_EVALUATE_EUCLIDEAN_DISTANCES = True
SHOULD_EVALUATE_EUCLIDEAN_DIST_ROC = True

if __name__ == '__main__':
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format_str)

    if SHOULD_TRAIN:
        train()
        torch.cuda.empty_cache()

    if SHOULD_EVALUATE:
        evaluate_model(should_save_accuracy=SHOULD_EVALUATE_ACCURACY,
                       should_save_accuracy_roc=SHOULD_EVALUATE_ACCURACY_ROC,
                       should_save_euclidean_distances=SHOULD_EVALUATE_EUCLIDEAN_DISTANCES,
                       should_save_euclidean_dist_roc=SHOULD_EVALUATE_EUCLIDEAN_DIST_ROC)


"""
from iris_dataset_processing.circle_crop import CircleCrop
from PIL import Image

if __name__ == '__main__':
    image = Image.open('data/oculariris-fc-left/s001_L_OffAngle_Fc_N10_001.jpg')
    circle_crop = CircleCrop(radius_percent_range=(0.8, 1.0), mode='outer')
    image = circle_crop(image)
    image.show()
    """
