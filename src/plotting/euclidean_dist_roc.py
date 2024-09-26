

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from common.output_reader import OutputReader
from common.output_writer import OutputWriter


def plot_euclidean_dist_roc(euclidean_dist_files: list[str],
                            title: str,
                            model_filebasename: str,
                            filename: str):
    log = logging.getLogger(__name__)
    log.info('Plotting euclidean distance ROC curve')
        
    fpr_rates = []
    tpr_rates = []
    thresholds_rates= []
    roc_labels = []
    
    output_reader = OutputReader()

    for datafile in euclidean_dist_files:
        log.info(f'Loading data from "{datafile}"')
        row_labels = output_reader.read_hd5_data(filename=datafile, model_filebasename=model_filebasename, dataset_name='row_labels')
        distance_matrix = output_reader.read_hd5_data(filename=datafile, model_filebasename=model_filebasename, dataset_name='euclidean_distance_matrix')

        distance_matrix = np.nan_to_num(distance_matrix, nan=0)

        log.info('Calculating intraclass and interclass distances')
        intraclass_distances = []
        interclass_distances = []
        
        for row in range(len(row_labels)):
            row_subject = row_labels[row].decode('utf-8').split('_')[0]
            for col in range(len(row_labels)):
                col_subject = row_labels[col].decode('utf-8').split('_')[0]
                if row_subject == col_subject:
                    intraclass_distances.append(distance_matrix[row, col])
                else:
                    interclass_distances.append(distance_matrix[row, col])
                    
        distances = intraclass_distances + interclass_distances
        distances = -np.array(distances)
        labels = [1]*len(intraclass_distances) + [0]*len(interclass_distances)
        
        fpr, tpr, thresholds = roc_curve(labels, distances)
        label = os.path.splitext(datafile)[0].replace('_percent', '%').removesuffix('_dataset_euclideandist').replace('_', ' ').title()

        fpr_rates.append(fpr)
        tpr_rates.append(tpr)
        thresholds_rates.append(thresholds)
        roc_labels.append(label)
    
    log.info('Building ROC curve')
    plt.figure()

    for i in range(len(fpr_rates)):
        plt.plot(fpr_rates[i], tpr_rates[i], label=labels[i])

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    #plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right")

    output_writer = OutputWriter()
    log.info(f'Saving accuracy roc graph to {output_writer.fig_dir}/{model_filebasename}/{filename}')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())

    plt.tight_layout()
    plt.clf()