# -*- coding: utf-8 -*-
"""
Module: euclidean_dist_histogram.py
Author: Daniel Tebor
Description: This module contains a function for plotting a histogram of the euclidean distances.
"""

import logging

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from common.output_writer import OutputWriter


def plot_euclidean_dist_histogram(row_labels: list[str],
                                  distance_matrix: np.ndarray,
                                  title: str,
                                  model_filebasename: str,
                                  filename: str):
    log = logging.getLogger(__name__)
    log.info('Plotting euclidean distance histogram')

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
    
    log.info('Building histogram')
    intraclass_bins = np.histogram_bin_edges(intraclass_distances, bins='auto')
    interclass_bins = np.histogram_bin_edges(interclass_distances, bins='auto')
    
    intraclass_hist, _ = cp.histogram(cp.asarray(intraclass_distances), density=True, bins=intraclass_bins)
    interclass_hist, _ = cp.histogram(cp.asarray(interclass_distances), density=True, bins=interclass_bins)

    intraclass_hist = cp.asnumpy(intraclass_hist)
    interclass_hist = cp.asnumpy(interclass_hist)
    
    plt.title(title)
    plt.bar(intraclass_bins[:-1], intraclass_hist, alpha=0.5, label='Intraclass', width=np.diff(intraclass_bins))
    plt.bar(interclass_bins[:-1], interclass_hist, alpha=0.5, label='Interclass', width=np.diff(interclass_bins))
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.xlim(0, 64)
    plt.ylim(0, 0.35)
    plt.legend()
    plt.margins(0)
    
    output_writer = OutputWriter()
    log.info(f'Saving euclidean distance histogram to "{output_writer.fig_dir}/{model_filebasename}/{filename}"')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())
    
    plt.clf()