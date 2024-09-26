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


def plot_roc(fpr_rates: list[np.array],
             tpr_rates: list[np.array],
             labels: list[str],
             model_filebasename: str,
             filename: str,
             title: str = None,
             legend_title: str = None,
             ylim: tuple[float, float] = (0.0, 1.0),
             should_mark_eer: bool = False,
             should_use_tight_layout: bool = False):
    log = logging.getLogger(__name__)
    log.info('Plotting accuracy roc graph')
    
    plt.figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF']

    for i in range(len(fpr_rates)):
        plt.plot(fpr_rates[i], tpr_rates[i], label=labels[i], color=colors[i % len(colors)])

        if should_mark_eer:
            eer = fpr_rates[i][np.nanargmin(np.abs((1 - tpr_rates[i]) - fpr_rates[i]))]
            plt.plot(eer, 1 - eer, 'o', color=colors[i % len(colors)])
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if title not in [None, '']:
        plt.title(title)
    plt.xlim(0.0, 1.0)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', title=legend_title)

    output_writer = OutputWriter()
    log.info(f'Saving accuracy roc graph to {output_writer.fig_dir}/{model_filebasename}/{filename}')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())

    if should_use_tight_layout:
        plt.tight_layout()
    plt.clf()