# -*- coding: utf-8 -*-
"""
Module: resnet50_merged_rgb_paired_left_and_right_oculariris.py
Author: Daniel Tebor
"""

import logging
import os

from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from common.output_reader import OutputReader
from common.output_writer import OutputWriter
from iris_dataset_processing.iris_dataset import IrisDataset
from iris_dataset_processing.deterministic_paired_iris_dataset import DeterministicPairedIrisDataset
from iris_dataset_processing.random_sample_paired_iris_dataset import RandomSamplePairedIrisDataset
from iris_dataset_processing.config.training_left_and_right_oculariris_dataset import *
from iris_dataset_processing.config.testing_left_and_right_oculariris_dataset import *
from iris_dataset_processing.config.validation_n50_thru_p50_left_and_right_oculariris_datasets import *
from models.cnn_model_factory import CNNModelFactory
from models.model_operators.cnn_model_operator import CNNModelOperator
from models.model_operators.cnn_model_operator_factory import CNNModelOperatorFactory, ModelOperatorName
from plotting.accuracy_bargraph import plot_accuracy_bargraph
from plotting.roc import plot_accuracy_roc
from plotting.euclidean_dist_histogram import plot_euclidean_dist_histogram
from plotting.euclidean_dist_roc import plot_euclidean_dist_roc


INPUT_SHAPE = (3, 224, 224)
MODELNAME = 'resnet50'
MODEL_FILENAME = f'{MODELNAME}_merged_rgb_{TRAINING_DERIVED_MODEL_DESCRIPTOR}.pt'
MODEL_FILEBASENAME = os.path.splitext(MODEL_FILENAME)[0]
LABEL_ENCODER_FILENAME = f'{MODEL_FILEBASENAME}_label_encoder.pkl'

def train():
    log = logging.getLogger(__name__)

    log.info('Creating subject complete dataset')
    subject_complete_dataset = IrisDataset(data_path=TRAINING_DATASET_PATH,
                                           cohorts=TRAINING_COHORTS,
                                           img_exclusion_flags=TRAINING_IMG_EXCLUSION_FLAGS,
                                           img_input_shape=INPUT_SHAPE)
    
    num_classes = subject_complete_dataset.num_classes
    label_encoder = subject_complete_dataset.label_encoder

    log.info(f'Creating {MODELNAME} model')

    cnn_model_factory = CNNModelFactory()
    model, criterion, optimizer = cnn_model_factory.create_model(modelname=MODELNAME, num_classes=num_classes)
    
    left_eye_training_dataset = IrisDataset(data_path=TRAINING_DATASET_PATH,
                                            cohorts=LEFT_EYE_TRAINING_COHORTS,
                                            img_exclusion_flags=TRAINING_IMG_EXCLUSION_FLAGS,
                                            img_input_shape=INPUT_SHAPE,
                                            label_encoder=label_encoder,
                                            should_use_augmentation=True)
    right_eye_training_dataset = IrisDataset(data_path=TRAINING_DATASET_PATH,
                                             cohorts=RIGHT_EYE_TRAINING_COHORTS,
                                             img_exclusion_flags=TRAINING_IMG_EXCLUSION_FLAGS,
                                             img_input_shape=INPUT_SHAPE,
                                             label_encoder=label_encoder,
                                             should_use_augmentation=True)
    paired_training_dataset = RandomSamplePairedIrisDataset(left_iris_dataset=left_eye_training_dataset, right_iris_dataset=right_eye_training_dataset, times_to_repeat_label=10)

    training_dataloader = DataLoader(dataset=paired_training_dataset,
                                     batch_size=TRAINING_BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=os.cpu_count())
    
    left_eye_testing_dataset = IrisDataset(data_path=TESTING_DATASET_PATH,
                                           cohorts=LEFT_EYE_TESTING_COHORTS,
                                           img_exclusion_flags=TESTING_IMG_EXCLUSION_FLAGS,
                                           img_input_shape=INPUT_SHAPE,
                                           label_encoder=label_encoder,
                                           should_use_augmentation=False)
    right_eye_testing_dataset = IrisDataset(data_path=TESTING_DATASET_PATH,
                                            cohorts=RIGHT_EYE_TESTING_COHORTS,
                                            img_exclusion_flags=TESTING_IMG_EXCLUSION_FLAGS,
                                            img_input_shape=INPUT_SHAPE,
                                            label_encoder=label_encoder,
                                            should_use_augmentation=False)
    paired_testing_dataset = DeterministicPairedIrisDataset(left_iris_dataset=left_eye_testing_dataset, right_iris_dataset=right_eye_testing_dataset)
    
    testing_dataloader = DataLoader(dataset=paired_testing_dataset,
                                    batch_size=TESTING_BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=os.cpu_count())

    model_operator_factory = CNNModelOperatorFactory(model=model, modelname=MODELNAME, num_output_classes=num_classes, criterion=criterion, optimizer=optimizer)
    model_operator = model_operator_factory.create_model_operator(ModelOperatorName.MERGED_RGB)
    epochs_taken, best_avg_loss = model_operator.train(training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
    
    should_save_model = input('Save model (y)? ').lower()
    if should_save_model != 'y':
        return
    
    output_writer = OutputWriter()

    log.info(f'Saving model to "{output_writer.model_dir}"')
    output_writer.write_model(filename=MODEL_FILENAME, model=model)

    log.info(f'Saving label_encoder to "{output_writer.label_encoder_dir}"')
    output_writer.write_label_encoder(filename=LABEL_ENCODER_FILENAME, label_encoder=label_encoder)

    log.info(f'Saving training metadata to "{output_writer.metadata_dir}/{MODEL_FILEBASENAME}"')
    img_paths = list(paired_training_dataset.left_iris_dataset.img_paths) + list(paired_training_dataset.right_iris_dataset.img_paths)
    imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
    training_metadata = {
        'cohorts': [cohort.value for cohort in TRAINING_COHORTS],
        'img_exclusion_flags': [flag.value for flag in TRAINING_IMG_EXCLUSION_FLAGS],
        'num_classes': num_classes,
        'batch_size': TRAINING_BATCH_SIZE,
        'epochs_taken': epochs_taken,
        'best_avg_loss': best_avg_loss,
        'label_encoder_filename': LABEL_ENCODER_FILENAME,
        'num_imgs': len(imgs),
        'imgs': imgs,
    }
    output_writer.write_metadata(model_filebasename=MODEL_FILEBASENAME, filename='training_metadata.json', metadata=training_metadata)

def _evaluate_accuracy(model_operator: CNNModelOperator,
                       label_encoder: LabelEncoder,
                       should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {MODEL_FILEBASENAME} accuracy')
    
    accuracies = {}
    output_writer = OutputWriter()

    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        left_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                       cohorts=config['left_eye_cohorts'],
                                       img_exclusion_flags=config['img_exclusion_flags'],
                                       img_input_shape=INPUT_SHAPE,
                                       label_encoder=label_encoder)
        right_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                        cohorts=config['right_eye_cohorts'],
                                        img_exclusion_flags=config['img_exclusion_flags'],
                                        img_input_shape=INPUT_SHAPE,
                                        label_encoder=label_encoder)
        paired_dataset = DeterministicPairedIrisDataset(left_iris_dataset=left_eye_dataset, right_iris_dataset=right_eye_dataset)
        
        dataloader = DataLoader(dataset=paired_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count())
        
        accuracy = model_operator.evaluate_model_accuracy(dataloader)
        accuracies[label.split('_')[0]] = accuracy
        log.info(f'{pretty_label} accuracy: {accuracy}')
    
        if not should_write_metadata:
            continue

        img_paths = list(paired_dataset.left_eye_img_paths) + list(paired_dataset.right_eye_img_paths)
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': LABEL_ENCODER_FILENAME,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{MODEL_FILEBASENAME}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=MODEL_FILEBASENAME,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    output_writer.write_data(model_filebasename=MODEL_FILEBASENAME,
                             filename='accuracy.json',
                             data=accuracies)
    
    plot_accuracy_bargraph(accuracies=accuracies,
                           title='Accuracy vs. Occlusion',
                           xlabel='Angle',
                           model_filebasename=MODEL_FILEBASENAME,
                           filename=f'accuracy_bargraph.png')
    
def _evaluate_roc(model_operator: CNNModelOperator,
                  label_encoder: LabelEncoder,
                  should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {MODEL_FILEBASENAME} accuracy roc')
    
    fpr_rates = []
    tpr_rates = []
    labels = []

    output_writer = OutputWriter()
    
    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        left_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                       cohorts=config['left_eye_cohorts'],
                                       img_exclusion_flags=config['img_exclusion_flags'],
                                       img_input_shape=INPUT_SHAPE,
                                       label_encoder=label_encoder)
        right_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                        cohorts=config['right_eye_cohorts'],
                                        img_exclusion_flags=config['img_exclusion_flags'],
                                        img_input_shape=INPUT_SHAPE,
                                        label_encoder=label_encoder)
        paired_dataset = DeterministicPairedIrisDataset(left_iris_dataset=left_eye_dataset, right_iris_dataset=right_eye_dataset)
        
        dataloader = DataLoader(dataset=paired_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count())

        fpr, tpr = model_operator.evaluate_model_roc(dataloader)
        fpr_rates.append(fpr)
        tpr_rates.append(tpr)
        labels.append(label.replace('_percent', '%').removesuffix('_dataset').replace('_', ' ').title())

        if not should_write_metadata:
            continue

        img_paths = list(paired_dataset.left_eye_img_paths) + list(paired_dataset.right_eye_img_paths)
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': LABEL_ENCODER_FILENAME,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{MODEL_FILEBASENAME}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=MODEL_FILEBASENAME,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    accuracy_roc_data = {
        'fpr_rates': [fpr.tolist() for fpr in fpr_rates],
        'tpr_rates': [tpr.tolist() for tpr in tpr_rates],
        'labels': labels
    }
    output_writer.write_data(model_filebasename=MODEL_FILEBASENAME,
                             filename='accuracy_roc.json',
                             data=accuracy_roc_data)
    
    plot_accuracy_roc(fpr_rates=fpr_rates,
                      tpr_rates=tpr_rates,
                      labels=labels,
                      title='Accuracy ROC Curve',
                      model_filebasename=MODEL_FILEBASENAME,
                      filename='accuracy_roc.png',
                      legend_title='Angle')
    
def _evaluate_euclidean_distances(model_operator: CNNModelOperator,
                                  label_encoder: LabelEncoder,
                                  should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {MODEL_FILEBASENAME} euclidean distances')

    output_writer = OutputWriter()

    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        left_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                       cohorts=config['left_eye_cohorts'],
                                       img_exclusion_flags=config['img_exclusion_flags'],
                                       img_input_shape=INPUT_SHAPE,
                                       label_encoder=label_encoder)
        right_eye_dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                                        cohorts=config['right_eye_cohorts'],
                                        img_exclusion_flags=config['img_exclusion_flags'],
                                        img_input_shape=INPUT_SHAPE,
                                        label_encoder=label_encoder)
        paired_dataset = DeterministicPairedIrisDataset(left_iris_dataset=left_eye_dataset, right_iris_dataset=right_eye_dataset)
        
        dataloader = DataLoader(dataset=paired_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True)
        
        filename = f'{label}_euclideandist.hdf5'
        model_operator.evaluate_model_euclidean_distances_and_save(dataloader=dataloader, model_filebasename=MODEL_FILEBASENAME, filename=filename)

        img_paths = list(paired_dataset.left_eye_img_paths) + list(paired_dataset.right_eye_img_paths)
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': LABEL_ENCODER_FILENAME,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        if not should_write_metadata:
            continue

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{MODEL_FILEBASENAME}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=MODEL_FILEBASENAME,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    output_reader = OutputReader()
    euclidean_dist_files = output_reader.get_data_files(model_filebasename=MODEL_FILEBASENAME)
    euclidean_dist_files = list(filter(lambda f: f.endswith('.hdf5'), euclidean_dist_files))

    for euclidean_distance_file in euclidean_dist_files:
        log.info(f'Loading euclidean distance data from "{output_reader}/{MODEL_FILEBASENAME}/{euclidean_distance_file}"')
        row_labels = output_reader.read_hd5_data(filename=euclidean_distance_file, model_filebasename=MODEL_FILEBASENAME, dataset_name='row_labels')
        distance_matrix = output_reader.read_hd5_data(filename=euclidean_distance_file, model_filebasename=MODEL_FILEBASENAME, dataset_name='euclidean_distance_matrix')

        title = os.path.splitext(euclidean_distance_file)[0].replace('_percent', '%').removesuffix('_dataset_euclideandist').replace('_', ' ').title()
        title = title + ' Euclidean Distance Histogram'
        plot_euclidean_dist_histogram(row_labels=row_labels,
                                      distance_matrix=distance_matrix,
                                      title=title,
                                      model_filebasename=MODEL_FILEBASENAME,
                                      filename=f'{os.path.splitext(euclidean_distance_file)[0]}_histogram.png')
        
def evaluate_model(should_save_accuracy: bool, should_save_roc: bool, should_save_euclidean_distances: bool):
    log = logging.getLogger(__name__)
    log.info(f'Loading model from {MODEL_FILENAME}')
    
    model_factory = CNNModelFactory()
    model, criterion, optimizer = model_factory.load_model(model_filename=MODEL_FILENAME)

    output_reader = OutputReader()
    label_encoder = output_reader.read_label_encoder(LABEL_ENCODER_FILENAME)

    model_training_metadata = output_reader.read_metadata(filename='training_metadata.json', model_filebasename=MODEL_FILEBASENAME)
    num_classes = model_training_metadata['num_classes']

    model_operator_factory = CNNModelOperatorFactory(model=model, modelname=MODELNAME, num_output_classes=num_classes, criterion=criterion, optimizer=optimizer)
    model_operator = model_operator_factory.create_model_operator(ModelOperatorName.MERGED_RGB)

    should_write_metadata = True

    if should_save_accuracy:
        _evaluate_accuracy(model_operator=model_operator,
                           label_encoder=label_encoder,
                           should_write_metadata=should_write_metadata)
        should_write_metadata = False
        
    torch.cuda.empty_cache()
        
    if should_save_roc:
        _evaluate_roc(model_operator=model_operator,
                      label_encoder=label_encoder,
                      should_write_metadata=should_write_metadata)
        should_write_metadata = False
        
    torch.cuda.empty_cache()

    if should_save_euclidean_distances:
        _evaluate_euclidean_distances(model_operator=model_operator,
                                      label_encoder=label_encoder,
                                      should_write_metadata=should_write_metadata)