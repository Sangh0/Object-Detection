import random
import numpy as np
import albumentations as A
from typing import *

if A.__version__ != '1.0.3':
    raise ValueError(f'The version of albumentations must be 1.0.3, not {A.__version__}')


class Augmentation(object):
    """
    Args:
        img_size: image size with tuple type
        horizontal_p: a probability for horizontal flip augmentation
        rotate_p: a probability for rotation augmentation
        clahe_p: a probability for CLAHE augmentation
        brightness_p: a probability for brightness in ColorJitter
        contrast_p: a probability for contrast in ColorJitter
        saturation_p: a probability for saturation in ColorJitter
    
    Returns:
        image: image array
        label: label array with class number and box coordinates
    """
    def __init__(
        self, 
        label_format: str='yolo',
        img_size: Tuple[int]=(416, 416), 
        horizontal_p: Optional[float]=0.5, 
        rotate_p: Optional[float]=0.5, 
        clahe_p: Optional[float]=0.3,
        brightness_p: Optional[float]=0.2,
        contrast_p: Optional[float]=0.2,
        saturation_p: Optional[float]=0.2,
    ):  
        assert label_format in ('coco', 'yolo')
        
        augmentations = [
            A.RandomResizedCrop(height=img_size[0], width=img_size[1]),
        ]

        if (horizontal_p is not None) and (type(horizontal_p) is float):
            augmentations.append(A.HorizontalFlip(p=horizontal_p))

        if (rotate_p is not None) and (type(rotate_p) is float):
            augmentations.append(A.Rotate(limit=40, p=rotate_p))

        if (clahe_p is not None) and (type(clahe_p) is float):
            augmentations.append(A.CLAHE(p=clahe_p))

        if brightness_p and contrast_p and saturation_p:
            augmentations.append(A.ColorJitter(brightness=brightness_p, contrast=contrast_p, saturation=saturation_p))

        self.transforms_ = A.Compose(
            augmentations, bbox_params=A.BboxParams(format=label_format, label_fields=['class_labels'])
        )

    def __call__(self, image, label, p=1.0):
        if random.random() < p:
            augmented = self.transforms_(image=image, bboxes=label[:, 1:], class_labels=label[:, 0])
            image = augmented['image']
            label = np.array([[c, *b] for c, b in zip(augmented['class_labels'], augmented['bboxes'])])
        return image, label