import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image


IRIS_IMAGE_FILENAME_LABEL_IDX = 0
IRIS_IMAGE_FILENAME_ANGLE_IDX = 4
P50_ANGLE = 'P50'
P40_ANGLE = 'P40'
P30_ANGLE = 'P30'
P20_ANGLE = 'P20'
P10_ANGLE = 'P10'
FRONTAL_ANGLE = 'P00'
N10_ANGLE = 'N10'
N20_ANGLE = 'N20'
N30_ANGLE = 'N30'
N40_ANGLE = 'N40'
N50_ANGLE = 'N50'
IRIS_IMAGE_ANGLES = [
    P50_ANGLE,
    P40_ANGLE,
    P30_ANGLE,
    P20_ANGLE,
    P10_ANGLE,
    FRONTAL_ANGLE,
    N10_ANGLE,
    N20_ANGLE,
    N30_ANGLE,
    N40_ANGLE,
    N50_ANGLE
]
IMAGE_MODE_RGB = 'RGB'
IMAGE_MODE_GRAY = 'L'
IMAGE_MODES = [
    IMAGE_MODE_RGB,
    IMAGE_MODE_GRAY
]

class IrisDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: transforms.Compose,
                 angles: list[str] = None,
                 label_encoder: LabelEncoder = None,
                 image_mode: str = IMAGE_MODE_RGB):
        assert os.path.exists(data_dir), f'Data directory does not exist: {data_dir}'
        if angles is not None:
            assert all(angle in IRIS_IMAGE_ANGLES for angle in angles), f'Invalid angle in angles: {angles}'
        assert image_mode in IMAGE_MODES, f'Invalid image mode: {image_mode}'
        
        self.data_dir = data_dir
        self.transform = transform
        self.image_mode = image_mode
        self.image_files = os.listdir(data_dir)
        
        image_filenames = []
        labels = []
        for image_file in self.image_files:
            angle = image_file.split('_')[IRIS_IMAGE_FILENAME_ANGLE_IDX]
            if angles is None or angle in angles:
                image_filenames.append(image_file)
                labels.append(image_file.split('_')[IRIS_IMAGE_FILENAME_LABEL_IDX])
        self.image_filenames = np.array(image_filenames)
        self.labels = np.array(labels)
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels)
        else:
            self.label_encoder = label_encoder
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        sample = Image.open(image_path).convert(self.image_mode)
        sample = self.transform(sample)
        
        encoded_label = self.encoded_labels[idx]
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)
        
        return sample, encoded_label
    
    def get_raw(self, idx) -> tuple[Image.Image, str]:
        sample_path = os.path.join(self.data_dir, self.image_filenames[idx])
        sample = Image.open(sample_path).convert(self.image_mode)
        
        label = self.labels[idx]
        
        return sample, label
    
def get_encoder_for_all_labels_in(data_dirs: list[str]) -> LabelEncoder:
    labels = []
    for data_dir in data_dirs:
        assert os.path.exists(data_dir), f'Data directory does not exist: {data_dir}'
        image_files = os.listdir(data_dir)
        for image_file in image_files:
            labels.append(image_file.split('_')[IRIS_IMAGE_FILENAME_LABEL_IDX])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    return label_encoder