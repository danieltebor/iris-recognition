# -*- coding: utf-8 -*-
"""
Module: output_writer.py
Author: Daniel Tebor
Description: This module contains a class for writing to the output directory.
"""

import io
import json
import logging
import os
import pickle
from typing import Any

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder


class OutputWriter:
    """
    A class for writing various types of output files related to a model.

    Attributes:
        base_dir (str): The base directory to write the output files to.

    Methods:
        __init__(self, base_dir='out'): Initializes an OutputWriter object with a base directory.
        write_data(self, model_name: str, filename: str, data: Any): Writes data to a JSON file.
        write_hdf5_data(self, model_name: str, filename: str, dataset_name: str, data: np.ndarray): Writes or appends a new hdf5 file.
        append_hdf5_data_slice(self, model_name: str, filename: str, dataset_name: str, data_slice: np.ndarray, start_idx: int): Appends a slice of data to an existing HDF5 dataset.
        write_fig(self, model_filename: str, filename: str, fig: plt.Figure): Save a matplotlib figure to a file.
        write_metadata(self, model_filename: str, filename: str, metadata: Any): Writes metadata to a JSON file.
        write_model(self, filename: str, file_object: IO): Writes a PyTorch model to a pt file.
        __del__(self): Closes the OutputWriter object.
    """

    def __init__(self, base_dir='./out'):
        self._base_dir = base_dir

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'fig'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'metadata'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'model'), exist_ok=True)

    def _write_file(self, filename: str, file_object: bytes, subdir: str = '', model_filebasename: str = None):
        out_path = os.path.join(self._base_dir, subdir)
        if model_filebasename is not None:
            out_path = os.path.join(out_path, model_filebasename)
        os.makedirs(out_path, exist_ok=True)
        file_path = os.path.join(out_path, filename)

        with open(file_path, 'wb') as f:
            f.write(file_object)

        log = logging.getLogger(__name__)
        log.info(f'Wrote {filename} to {out_path}')

    def write_data(self, model_filebasename: str, filename: str, data: Any):
        """
        Write data to a JSON file.

        Args:
            model_name (str): The name of the model.
            filename (str): The name of the file to write the data to.
            data (Any): The data to write to the file.

        Raises:
            ValueError: If the filename does not end with ".json".
        """

        if not filename.endswith('.json'):
            raise ValueError(f'Data filename must end with ".json".')

        file_object = json.dumps(data, indent=4).encode('utf-8')
        self._write_file(filename, file_object, 'data', model_filebasename)

    def write_hdf5_data(self, model_filebasename: str, filename: str, dataset_name: str, data: np.ndarray):
        """
        Writes or appends a new hdf5 file. Adds a new dataset to the file or overwrites an existing dataset.

        Args:
            model_name (str): The name of the model.
            filename (str): The name of the HDF5 file to write to.
            dataset_name (str): The name of the dataset to create or overwrite in the HDF5 file.
            data (np.ndarray): The numpy array to write to the dataset.

        Returns:
            None

        Raises:
            ValueError: If the filename does not end with ".hdf5".
        """

        if not filename.endswith('.hdf5'):
            raise ValueError(f'HDF5 filename must end with ".hdf5".')

        out_path = os.path.join(self._base_dir, 'data')
        out_path = os.path.join(out_path, model_filebasename)
        os.makedirs(out_path, exist_ok=True)
        file_path = os.path.join(out_path, filename)

        with h5py.File(file_path, 'a') as f:
            # Overwrite dataset if it already exists.
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=data, compression="gzip", compression_opts=9)

        log = logging.getLogger(__name__)
        log.info(f'Wrote {dataset_name} dataset to {filename}')

    def append_hdf5_data_slice(self, model_filebasename: str, filename: str, dataset_name: str, data_slice: np.ndarray, start_idx: int):
        """
        Appends a slice of data to an existing HDF5 dataset.

        Args:
            model_name (str): The name of the model.
            filename (str): The name of the HDF5 file.
            dataset_name (str): The name of the dataset to append to.
            data_slice (np.ndarray): The slice of data to append.
            start_idx (int): The starting index to append the data slice to.

        Raises:
            ValueError: If the file or dataset does not exist, or if the data slice cannot be appended to the dataset.
        """

        out_path = os.path.join(self._base_dir, 'data')
        out_path = os.path.join(out_path, model_filebasename)
        os.makedirs(out_path, exist_ok=True)
        file_path = os.path.join(out_path, filename)

        if not os.path.exists(file_path):
            raise ValueError(f'File {file_path} does not exist.')

        with h5py.File(file_path, 'a') as f:
            if dataset_name not in f:
                raise ValueError(f'Dataset {dataset_name} does not exist in file {filename}.')
            else:
                end_idx = start_idx + len(data_slice)
                try:
                    f[dataset_name][start_idx:end_idx] = data_slice
                except ValueError:
                    raise ValueError(f'Cannot append data slice of length {len(data_slice)} at start index {start_idx} '
                                     f'to dataset {dataset_name} with shape {f[dataset_name].shape}.')
        
        log = logging.getLogger(__name__)
        log.info(f'Appended data slice to the {dataset_name} dataset in {filename} at index {start_idx}')
    
    def write_fig(self, model_filebasename: str, filename: str, fig: plt.Figure):
        """
        Save a matplotlib figure to a file.

        Args:
            model_filename (str): The name of the model file being used.
            filename (str): The name of the file to save the figure to.
            fig (matplotlib.figure.Figure): The figure to save.

        Raises:
            ValueError: If the filename does not end with ".png".
        """

        if not filename.endswith('.png'):
            raise ValueError(f'Figure filename must end with ".png".')
        
        file_object = io.BytesIO()
        fig.canvas.print_png(file_object)
        file_object.seek(0)
        self._write_file(filename, file_object.read(), 'fig', model_filebasename)

    def write_label_encoder(self, filename: str, label_encoder: LabelEncoder):
        """
        Writes a LabelEncoder object to a file.

        Args:
            filename (str): The name of the file to write the LabelEncoder object to.
            label_encoder (LabelEncoder): The LabelEncoder object to write to the file.

        Raises:
            ValueError: If the filename does not end with ".pkl".
        """

        if not filename.endswith('.pkl'):
            raise ValueError(f'Label encoder filename must end with ".pkl".')

        file_object = io.BytesIO()
        pickle.dump(label_encoder, file_object)
        file_object.seek(0)
        self._write_file(filename, file_object.read(), 'label_encoder')

    def write_metadata(self, model_filebasename: str, filename: str, metadata: Any):
        """
        Writes metadata to a JSON file.

        Args:
            model_filename (str): The name of the model file.
            filename (str): The name of the metadata file.
            metadata (Any): The metadata to be written to the file.

        Raises:
            ValueError: If the metadata filename does not end with ".json".
        """

        if not filename.endswith('.json'):
            raise ValueError(f'Metadata filename must end with ".json".')

        file_object = json.dumps(metadata, indent=4).encode('utf-8')
        self._write_file(filename, file_object, 'metadata', model_filebasename)

    def write_model(self, filename: str, model: nn.Module):
        """
        Saves the state dictionary of a PyTorch model to a file with the given filename.
        
        Args:
            filename (str): The name of the file to save the model to. Must end with ".pt".
            model (nn.Module): The PyTorch model to save.
        
        Raises:
            ValueError: If the filename does not end with ".pt".
        """
        
        if not filename.endswith('.pt'):
            raise ValueError(f'Model filename must end with ".pt".')
        
        file_object = io.BytesIO()
        torch.save(model.state_dict(), file_object)
        file_object.seek(0)
        self._write_file(filename, file_object.read(), 'model')

    @property
    def out_dir(self) -> str:
        return self._base_dir
    
    @property
    def data_dir(self) -> str:
        return self._base_dir + '/data'
    
    @property
    def fig_dir(self) -> str:
        return self._base_dir + '/fig'
    
    @property
    def label_encoder_dir(self) -> str:
        return self._base_dir + '/label_encoder'
    
    @property
    def metadata_dir(self) -> str:
        return self._base_dir + '/metadata'
    
    @property
    def model_dir(self) -> str:
        return self._base_dir + '/model'