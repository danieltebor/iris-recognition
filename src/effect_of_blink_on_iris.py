# -*- coding: utf-8 -*-
"""
Module: effect_of_blink_on_iris.py
Author: Daniel Tebor
Description: This module contains a script for training a model to evaluate
             the effect of blinking on iris recognition.
"""

import logging
import os

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from common.output_reader import OutputReader
from common.output_writer import OutputWriter
from iris_dataset_processing.iris_dataset import IrisDataset
from iris_dataset_processing.config.subject_complete_occlusion_dataset import *
#from iris_dataset_processing.config.training_original_occlusion_distribution_dataset import *
#from iris_dataset_processing.config.training_original_plus_0_to_30_percent_synthetic_occlusion import *
#from iris_dataset_processing.config.training_original_plus_0_to_60_percent_synthetic_occlusion import *
from iris_dataset_processing.config.training_original_plus_0_to_90_percent_synthetic_occlusion import *
from iris_dataset_processing.config.testing_0_to_90_percent_occlusion_dataset import *
from iris_dataset_processing.config.validation_0_thru_90_percent_occlusion_datasets import *
from models.cnn_model_factory import CNNModelFactory, ModelName
from models.cnn_model_operator import CNNModelOperator
from plotting.accuracy_bargraph import plot_accuracy_bargraph
from plotting.accuracy_roc import plot_accuracy_roc
from plotting.euclidean_dist_histogram import plot_euclidean_dist_histogram
from plotting.euclidean_dist_roc import plot_euclidean_dist_roc


def train():
    log = logging.getLogger(__name__)

    log.info('Creating subject complete dataset')
    subject_complete_dataset = IrisDataset(data_path=SUBJECT_COMPLETE_OCCLUSION_DATASET_PATH,
                                           cohorts=SUBJECT_COMPLETE_COHORTS,
                                           img_exclusion_flags=SUBJECT_COMPLETE_IMG_EXCLUSION_FLAGS,
                                           img_input_shape=(3, 224, 224))
    
    num_classes = subject_complete_dataset.num_classes
    label_encoder = subject_complete_dataset.label_encoder

    models = [model_name.value for model_name in ModelName]
    print(f'Models:')
    for i, model_name in enumerate(models):
        print(f'{i + 1}: {model_name}')

    chosen_model = None
    while True:
        try:
            chosen_model = int(input(f'Select a model (1-{len(models)}): '))
            if chosen_model < 1 or chosen_model > len(models):
                print('Invalid input. Try again.')
                continue
            break
        except ValueError:
            print('Invalid input. Try again.')

    model_name = models[chosen_model - 1]
    log.info(f'Creating {model_name} model')

    cnn_model_factory = CNNModelFactory()
    model, criterion, optimizer = cnn_model_factory.create_model(model_name=model_name, num_classes=num_classes)
    input_shape = cnn_model_factory.get_model_input_shape(model_name=model_name)
    
    training_dataset = IrisDataset(data_path=TRAINING_DATASET_PATH,
                                   cohorts=TRAINING_COHORTS,
                                   img_exclusion_flags=TRAINING_IMG_EXCLUSION_FLAGS,
                                   img_input_shape=input_shape,
                                   label_encoder=label_encoder,
                                   should_use_augmentation=True)
    training_dataloader = DataLoader(dataset=training_dataset,
                                     batch_size=TRAINING_BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=os.cpu_count(),
                                     pin_memory=True)
    
    testing_dataset = IrisDataset(data_path=TESTING_DATASET_PATH,
                                  cohorts=TESTING_COHORTS,
                                  img_exclusion_flags=TESTING_IMG_EXCLUSION_FLAGS,
                                  img_input_shape=input_shape,
                                  label_encoder=label_encoder,
                                  should_use_augmentation=False)
    testing_dataloader = DataLoader(dataset=testing_dataset,
                                    batch_size=TESTING_BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True)

    model_operator = CNNModelOperator(model=model, model_name=models[chosen_model - 1], num_output_classes=num_classes, criterion=criterion, optimizer=optimizer)
    epochs_taken, best_avg_loss = model_operator.train(training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
    
    should_save_model = input('Save model (y)? ').lower()
    if should_save_model != 'y':
        return
    
    output_writer = OutputWriter()

    log.info(f'Saving model to "{output_writer.model_dir}"')
    model_filename = f'{model_name}_{TRAINING_DERIVED_MODEL_DESCRIPTOR}.pt'
    model_filebasename = os.path.splitext(model_filename)[0]
    output_writer.write_model(filename=model_filename, model=model)

    log.info(f'Saving label_encoder to "{output_writer.label_encoder_dir}"')
    label_encoder_filename = f'{model_filebasename}_label_encoder.pkl'
    output_writer.write_label_encoder(filename=label_encoder_filename, label_encoder=label_encoder)

    log.info(f'Saving training metadata to "{output_writer.metadata_dir}/{model_filebasename}"')
    img_paths = training_dataset.img_paths
    imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
    training_metadata = {
        'cohorts': [cohort.value for cohort in TRAINING_COHORTS],
        'img_exclusion_flags': [flag.value for flag in TRAINING_IMG_EXCLUSION_FLAGS],
        'num_classes': num_classes,
        'batch_size': TRAINING_BATCH_SIZE,
        'epochs_taken': epochs_taken,
        'best_avg_loss': best_avg_loss,
        'label_encoder_filename': label_encoder_filename,
        'num_imgs': len(imgs),
        'imgs': imgs,
    }
    output_writer.write_metadata(model_filebasename=model_filebasename, filename='training_metadata.json', metadata=training_metadata)

def evaluate_model_accuracy(model_operator: CNNModelOperator,
                            model_filebasename: str,
                            model_input_shape: (int, int, int),
                            label_encoder: LabelEncoder,
                            label_encoder_filename: str,
                            should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {model_filebasename} accuracy')
    
    accuracies = {}
    output_writer = OutputWriter()

    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                              cohorts=config['cohorts'],
                              img_exclusion_flags=config['img_exclusion_flags'],
                              img_input_shape=model_input_shape,
                              label_encoder=label_encoder)
        
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True)
        
        accuracy = model_operator.evaluate_model_accuracy(dataloader)
        accuracies[label.split('_')[0]] = accuracy
        log.info(f'{pretty_label} accuracy: {accuracy}')
    
        if not should_write_metadata:
            continue

        img_paths = dataset.img_paths
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': label_encoder_filename,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{model_filebasename}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=model_filebasename,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    output_writer.write_data(model_filebasename=model_filebasename,
                             filename='accuracy.json',
                             data=accuracies)
    
    plot_accuracy_bargraph(accuracies=accuracies,
                           title='Accuracy vs. Occlusion',
                           xlabel='Occlusion (%)',
                           model_filebasename=model_filebasename,
                           filename=f'accuracy_bargraph.png')


def evaluate_model_accuracy_roc(model_operator: CNNModelOperator,
                                model_filebasename: str,
                                model_input_shape: (int, int, int),
                                label_encoder: LabelEncoder,
                                label_encoder_filename: str,
                                should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {model_filebasename} accuracy roc')
    
    fpr_rates = []
    tpr_rates = []
    labels = []

    output_writer = OutputWriter()
    
    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                              cohorts=config['cohorts'],
                              img_exclusion_flags=config['img_exclusion_flags'],
                              img_input_shape=model_input_shape,
                              label_encoder=label_encoder)
        
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True)

        fpr, tpr = model_operator.evaluate_model_roc(dataloader)
        fpr_rates.append(fpr)
        tpr_rates.append(tpr)
        labels.append(label.replace('_percent', '%').removesuffix('_dataset').replace('_', ' ').title())

        if not should_write_metadata:
            continue

        img_paths = dataset.img_paths
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': label_encoder_filename,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{model_filebasename}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=model_filebasename,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    accuracy_roc_data = {
        'fpr_rates': [fpr.tolist() for fpr in fpr_rates],
        'tpr_rates': [tpr.tolist() for tpr in tpr_rates],
        'labels': labels
    }
    output_writer.write_data(model_filebasename=model_filebasename,
                             filename='accuracy_roc.json',
                             data=accuracy_roc_data)
    
    plot_accuracy_roc(fpr_rates=fpr_rates,
                      tpr_rates=tpr_rates,
                      labels=labels,
                      title='Accuracy ROC Curve',
                      model_filebasename=model_filebasename,
                      filename='accuracy_roc.png')

def evaluate_euclidean_distances(model_operator: CNNModelOperator,
                                 model_filebasename: str,
                                 model_input_shape: (int, int, int),
                                 label_encoder: LabelEncoder,
                                 label_encoder_filename: str,
                                 should_write_metadata: bool = True):
    log = logging.getLogger(__name__)
    log.info(f'Evaluating {model_filebasename} euclidean distances')

    output_writer = OutputWriter()

    for label, config in VALIDATION_CONFIGS.items():
        pretty_label = label.replace('_', ' ').title()

        log.info(f'Validating with {pretty_label} configuration')

        dataset = IrisDataset(data_path=VALIDATION_DATASET_PATH,
                              cohorts=config['cohorts'],
                              img_exclusion_flags=config['img_exclusion_flags'],
                              img_input_shape=model_input_shape,
                              label_encoder=label_encoder)
        
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=os.cpu_count(),
                                pin_memory=True)
        
        filename = f'{label}_euclideandist.hdf5'
        model_operator.evaluate_model_euclidean_distances_and_save(dataloader=dataloader, model_filebasename=model_filebasename, filename=filename)

        img_paths = dataset.img_paths
        imgs = [os.path.join(os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)) for path in img_paths]
        validation_metadata = {'cohorts': [cohort.value for cohort in config['cohorts']],
                               'img_exclusion_flags': [flag.value for flag in config['img_exclusion_flags']],
                               'batch_size': config['batch_size'],
                               'label_encoder_filename': label_encoder_filename,
                               'num_imgs': len(imgs),
                               'imgs': imgs}

        if not should_write_metadata:
            continue

        log.info(f'Saving validation metadata to "{output_writer.metadata_dir}/{model_filebasename}/{label}_validation_metadata.json"')
        output_writer.write_metadata(model_filebasename=model_filebasename,
                                     filename=f'{label}_validation_metadata.json',
                                     metadata=validation_metadata)

    output_reader = OutputReader()
    euclidean_dist_files = output_reader.get_data_files(model_filebasename=model_filebasename)
    euclidean_dist_files = list(filter(lambda f: f.endswith('.hdf5'), euclidean_dist_files))

    for euclidean_distance_file in euclidean_dist_files:
        log.info(f'Loading euclidean distance data from "{output_reader}/{model_filebasename}/{euclidean_distance_file}"')
        row_labels = output_reader.read_hd5_data(filename=euclidean_distance_file, model_filebasename=model_filebasename, dataset_name='row_labels')
        distance_matrix = output_reader.read_hd5_data(filename=euclidean_distance_file, model_filebasename=model_filebasename, dataset_name='euclidean_distance_matrix')

        title = os.path.splitext(euclidean_distance_file)[0].replace('_percent', '%').removesuffix('_dataset_euclideandist').replace('_', ' ').title()
        title = title + ' Euclidean Distance Histogram'
        plot_euclidean_dist_histogram(row_labels=row_labels,
                                      distance_matrix=distance_matrix,
                                      title=title,
                                      model_filebasename=model_filebasename,
                                      filename=f'{os.path.splitext(euclidean_distance_file)[0]}_histogram.png')
        
    # plot_euclidean_dist_roc(euclidean_dist_files=euclidean_dist_files,
    #                         title='Euclidean Distance ROC Curve',
    #                         model_filebasename=model_filebasename,
    #                         filename=f'euclidean_distance_roc.png')

def load_model():
    print('Saved models:')

    output_reader = OutputReader()
    models = output_reader.get_model_files()
    i = 1
    for model in models:
        print(f'{i}: {model}')
        i += 1
    
    while True:
        try:
            model_to_load = int(input(f'Select a model (1-{i - 1}): '))
            if model_to_load < 1 or model_to_load > i - 1:
                print('Invalid input. Try again.')
                continue
            break
        except ValueError:
            print('Invalid input. Try again.')
    
    model_filename = models[model_to_load - 1]
    model_filebasename = os.path.splitext(model_filename)[0]
    model_name = model_filename.split('_')[0]

    log = logging.getLogger(__name__)
    log.info(f'Loading model from {model_filename}')
    
    model_factory = CNNModelFactory()
    model, criterion, optimizer = model_factory.load_model(model_filename=model_filename)
    input_shape = model_factory.get_model_input_shape(model_name=model_name)

    label_encoder_filename = model_filebasename + '_label_encoder.pkl'
    label_encoder = output_reader.read_label_encoder(label_encoder_filename)

    model_training_metadata = output_reader.read_metadata(filename='training_metadata.json', model_filebasename=model_filebasename)
    num_classes = model_training_metadata['num_classes']

    model_operator = CNNModelOperator(model=model, model_name=model_name, num_output_classes=num_classes, criterion=criterion, optimizer=optimizer)

    should_save_vparams = input('Do accuracy validation (y)? ').lower()
    should_save_roc = input('Do ROC evaluation (y)? ').lower()
    should_save_distances = input('Do euclidean distances evalauation (y)? ').lower()
    should_write_metadata = True
    
    if should_save_vparams == 'y':
        evaluate_model_accuracy(model_operator=model_operator,
                                model_filebasename=model_filebasename,
                                model_input_shape=input_shape,
                                label_encoder=label_encoder,
                                label_encoder_filename=label_encoder_filename)
        should_write_metadata = False
    
    if should_save_roc == 'y':
        evaluate_model_accuracy_roc(model_operator=model_operator,
                                    model_filebasename=model_filebasename,
                                    model_input_shape=input_shape,
                                    label_encoder=label_encoder,
                                    label_encoder_filename=label_encoder_filename,
                                    should_write_metadata=should_write_metadata)
        should_write_metadata = False
    
    if should_save_distances == 'y':
        evaluate_euclidean_distances(model_operator=model_operator,
                                     model_filebasename=model_filebasename,
                                     model_input_shape=input_shape,
                                     label_encoder=label_encoder,
                                     label_encoder_filename=label_encoder_filename,
                                     should_write_metadata=should_write_metadata)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = None
    operation = input('Train, load model, or regenerate graphs: (t/l/r)? ').lower()
    if operation == 't':
        model = train()
    elif operation == 'l':
        model = load_model()
    elif operation == 'r':
        pass