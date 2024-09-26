# -*- coding: utf-8 -*-
"""
Module: accuracy_bargraph.py
Author: Daniel Tebor
Description: This module contains a function for plotting a bar graph of the accuracy of each model.
"""

import logging

import matplotlib.pyplot as plt

from common.output_writer import OutputWriter


def plot_accuracy_bargraph(accuracies: dict[str, float],
                           xlabel: str,
                           model_filebasename: str,
                           filename: str,
                           title: str = None,
                           should_use_tight_layout: bool = False):
    log = logging.getLogger(__name__)
    log.info('Plotting accuracy bar graph')

    plt.bar(accuracies.keys(), accuracies.values())
    if title:
        plt.title(title)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    output_writer = OutputWriter()
    log.info(f'Saving accuracy bar graph to {output_writer.fig_dir}/{model_filebasename}/{filename}')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())

    if should_use_tight_layout:
        plt.tight_layout()
    plt.clf()