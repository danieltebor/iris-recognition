import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from iris.datasets.iris_dataset import *


class IrisTripletDataset(IrisDataset):
    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose,
        angles: list[str] = None,
        image_mode: str = IMAGE_MODE_RGB,
        angle_correction: bool = False,
        is_deterministic: bool = False,
        max_ancher_neg_angle_diff: int = None
    ):
        super().__init__(data_dir, transform, angles=angles, image_mode=image_mode, angle_correction=angle_correction)
        
        self.is_deterministic = is_deterministic
        self.max_ancher_neg_angle_diff = max_ancher_neg_angle_diff
        
        label_to_idx_mapping = {}
        for idx, label in enumerate(self.encoded_labels):
            label_to_idx_mapping.setdefault(label, []).append(idx)
        self.label_to_idx_mapping = {label: np.array(indices) for label, indices in label_to_idx_mapping.items()}
        
        if self.is_deterministic:
            self.rng = np.random.RandomState(42) # The answer to the universe or something.
            self.triplets = self._precompute_deterministic_triplets()
        else:
            self.rng = np.random.RandomState()
        
    def _precompute_deterministic_triplets(self):
        triplets = []
        for idx in range(len(self)):
            anchor_filename = self.image_filenames[idx]
            anchor_angle = self._get_angle_from_image_filename(anchor_filename)
            anchor_label = self.encoded_labels[idx]
            
            positive_idx_candidates = self.label_to_idx_mapping[anchor_label]
            positive_indices = [i for i in positive_idx_candidates if i != idx]
            positive_idx = self.rng.choice(positive_indices)
            
            negative_labels = [label for label in self.label_to_idx_mapping.keys() if label != anchor_label]
            negative_label = self.rng.choice(negative_labels)
            negative_idx_candidates = self.label_to_idx_mapping[negative_label]
            max_diff = self.max_ancher_neg_angle_diff if self.max_ancher_neg_angle_diff is not None else 180
            negative_idx_candidates = [i for i in negative_idx_candidates if
                                       abs(self._get_angle_from_image_filename(self.image_filenames[i]) - anchor_angle) <= max_diff]
            negative_idx = self.rng.choice(negative_idx_candidates)
            
            triplets.append(np.array([idx, positive_idx, negative_idx]))
        
        return np.array(triplets)
        
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_deterministic:
            anchor_idx, positive_idx, negative_idx = self.triplets[idx]
            anchor_filename = self.image_filenames[anchor_idx]
            anchor_path = os.path.join(self.data_dir, anchor_filename)
            positive_path = os.path.join(self.data_dir, self.image_filenames[positive_idx])
            negative_path = os.path.join(self.data_dir, self.image_filenames[negative_idx])
        else:
            anchor_filename = self.image_filenames[idx]
            anchor_angle = self._get_angle_from_image_filename(anchor_filename)
            anchor_label = self.encoded_labels[idx]
            anchor_path = os.path.join(self.data_dir, anchor_filename)
        
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = self.rng.choice(self.label_to_idx_mapping[anchor_label])
            positive_path = os.path.join(self.data_dir, self.image_filenames[positive_idx])
        
            negative_label = anchor_label
            while negative_label == anchor_label:
                negative_label = self.rng.choice(self.encoded_labels)
            negative_idx_candidates = self.label_to_idx_mapping[negative_label]
            max_diff = self.max_ancher_neg_angle_diff if self.max_ancher_neg_angle_diff is not None else 180
            negative_idx_candidates = [i for i in negative_idx_candidates if
                                       abs(self._get_angle_from_image_filename(self.image_filenames[i]) - anchor_angle) <= max_diff]
            negative_idx = self.rng.choice(negative_idx_candidates)
            negative_path = os.path.join(self.data_dir, self.image_filenames[negative_idx])

        anchor_sample = Image.open(anchor_path).convert(self.image_mode)
        positive_sample = Image.open(positive_path).convert(self.image_mode)
        negative_sample = Image.open(negative_path).convert(self.image_mode)
        
        if self.angle_correction:
            anchor_angle = anchor_filename.split('_')[IRIS_IMAGE_FILENAME_ANGLE_IDX]
            anchor_sample = self._apply_angle_correction(anchor_sample, anchor_angle)
            positive_angle = self.image_filenames[positive_idx].split('_')[IRIS_IMAGE_FILENAME_ANGLE_IDX]
            positive_sample = self._apply_angle_correction(positive_sample, positive_angle)
            negative_angle = self.image_filenames[negative_idx].split('_')[IRIS_IMAGE_FILENAME_ANGLE_IDX]
            negative_sample = self._apply_angle_correction(negative_sample, negative_angle)

        anchor_sample = self.transform(anchor_sample)
        positive_sample = self.transform(positive_sample)
        negative_sample = self.transform(negative_sample)
        
        return anchor_sample, positive_sample, negative_sample
    
    def _get_angle_from_image_filename(self, image_filename: str) -> int:
        angle = image_filename.split('_')[IRIS_IMAGE_FILENAME_ANGLE_IDX]
        if angle.startswith('N'):
            return int(angle[1:]) * -1
        else:
            return int(angle[1:])