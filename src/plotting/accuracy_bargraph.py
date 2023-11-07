# -*- coding: utf-8 -*-
"""
Module: accuracy_bargraph.py
Author: Daniel Tebor
Description: This module contains a function for plotting a bar graph of the accuracy of each model.
"""

import logging

import matplotlib.pyplot as plt

from common.output_writer import OutputWriter


def plot_accuracy_bargraph(accuracies: dict[str, float], title: str, xlabel: str, model_filebasename: str, filename: str):
    log = logging.getLogger(__name__)
    log.info('Plotting accuracy bar graph')

    plt.bar(accuracies.keys(), accuracies.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    output_writer = OutputWriter()
    log.info(f'Saving accuracy bar graph to {output_writer.fig_dir}/{model_filebasename}/{filename}')
    output_writer.write_fig(model_filebasename, filename, plt.gcf())

    plt.clf()