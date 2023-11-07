# -*- coding: utf-8 -*-
"""
Module: accuracy_roc.py
Author: Daniel Tebor
Description: This module contains a function for plotting a roc graph of the accuracy of each model.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from common.output_writer import OutputWriter


def plot_accuracy_roc(fpr_rates: list[np.array],
                      tpr_rates: list[np.array],
                      labels: list[str],
                      title: str,
                      model_filebasename: str,
                      filename: str):
    log = logging.getLogger(__name__)
    log.info('Plotting accuracy roc graph')
    
    plt.figure()

    for i in range(len(fpr_rates)):
        plt.plot(fpr_rates[i], tpr_rates[i], label=labels[i])
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    output_writer = OutputWriter()
    log.info(f'Saving accuracy roc graph to {output_writer.fig_dir}/{model_filebasename}/{filename}')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())

    plt.clf()