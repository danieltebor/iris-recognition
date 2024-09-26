from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class CircleCrop(object):
    def __init__(self, radius_percent_range: Tuple[float, float]=(0.9, 1.0), mode: str='outer'):
        assert 0.0 <= radius_percent_range[0] < radius_percent_range[1] <= 1.0
        self._radius_percent_range = radius_percent_range
        self._mode = mode

    def __call__(self, image: Image):
        # Convert to numpy array
        image = np.array(image)

        # Get the center of the image
        center = (image.shape[1] // 2, image.shape[0] // 2)

        # Get random value between the range
        radius_percent = np.random.uniform(*self._radius_percent_range)

        # Calculate the radius
        radius = int(min(center) * radius_percent)

        # Create a mask
        mask = np.zeros_like(image)
        cv2.circle(mask, center, radius, (1, 1, 1), -1)

        if self._mode == 'inner':
            mask = 1 - mask

        # Apply the mask
        result = image * mask

        # Convert back to PIL image
        result = Image.fromarray(result)

        return result