import yaml
import cv2
import numpy as np
from glob import glob
from PIL import Image
from typing import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .augmentation import Augmentation


class ObstacleDetectionDataset(Dataset):
    """
    Args:
        path: a dataset folder directory
        subset: train, valid or test set
        img_size: image size with tuple type
        horizontal_p: a probability for horizontal flip augmentation
        rotate_p: a probability for rotation augmentation
        clahe_p: a probability for CLAHE augmentation
        brightness_p: a probability for brightness in ColorJitter
        contrast_p: a probability for contrast in ColorJitter
        saturation_p: a probability for saturation in ColorJitter
    """
    def __init__(
        self,
        path: str,
        subset: str='train',
        img_size: Tuple[int]=(416,416),
        transforms_: Optional[bool]=None,
        horizontal_p: Optional[float]=0.5, 
        rotate_p: Optional[float]=0.5, 
        clahe_p: Optional[float]=0.3,
        brightness_p: Optional[float]=0.2,
        contrast_p: Optional[float]=0.2,
        saturation_p: Optional[float]=0.2,
    ):
        assert subset in ('train', 'valid', 'test'), \
            f'{subset} does not exists'
        self.subset = subset
        self.img_size = img_size
        self.normalized_label = normalized_label

        self.image_files = read_yaml_file(path, subset)
        self.label_files = [
            file.replace('images', 'labels').replace('.jpg', '.txt') for file in self.image_files
        ]
        
        assert len(self.image_files) == len(self.label_files), \
            f'The number of images {len(self.image_files)} and labels {len(self.label_files)} does not match'
        print(f'The number of {subset} dataset is {len(self.image_files)}')

        self.transforms_ = Augmentation(
            label_format='yolo',
            img_size=img_size,
            horizontal_p=horizontal_p,
            rotate_p=rotate_p,
            clahe_p=clahe_p,
            brightness_p=brightness_p,
            contrast_p=contrast_p,
            saturation_p=saturation_p,
        ) if transforms_ is not None else None
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = load_img_file(self.image_files[idx])
        lab = load_txt_file(self.label_files[idx])
        
        img, lab = letterbox(img, lab, self.img_size)

        if self.transforms_ is not None:
            img, lab = self.transforms_(image=img, label=lab)

        img = img.transpose(2, 0, 1) # last channel to first channel for torch tensor
        img = np.ascontiguousarray(img)
        
        labels_out = np.zeros((len(lab), 6))
        labels_out[:, 1] = lab[:, 0]
        labels_out[:, 2:] = lab[:, 1:]
        
        return torch.from_numpy(img/255.), torch.from_numpy(labels_out), self.image_files[idx]
        
    @staticmethod
    def collate_fn(batch):
        images, labels, paths = zip(*batch)
        path_list = []
        for i, label in enumerate(labels):
            label[:, 0] = i
            path_list.append(paths[i].split('/')[-1])
        return torch.stack(images, 0), torch.cat(labels, 0), path_list

    
def read_yaml_file(yaml_file, subset):
    """
    Args:
        yaml_file: a directory of yaml file 

    Returns:
        file_list: a image files list of subset data
    """
    with open(yaml_file, 'r') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)

    with open(file[subset], 'r') as f:
        file_list = [name.rstrip() for name in f.readlines()]

    return file_list


def load_img_file(path):
    """
    Args:
        path: a directory of image file

    Returns:
        img: a tensor of images with RGB channels
    """
    img = cv2.imread(path)
    img = img[:, :, ::-1] # BGR to RGB
    return img


def load_txt_file(path):
    """
    Args:
        path: a directory of text file (yolo format label)

    Returns:
        labels: np.array([[class, center x, center y, width, height] * number of labels])
    """
    with open(path, 'r') as f:
        labels = [x.split() for x in f.read().strip().splitlines()]
        if not labels: # empty label
            return np.zeros((0, 5), dtype=np.float32)
        classes = np.array([x[0] for x in labels], dtype=np.float32)
        bboxes = [np.array(x[1:], dtype=np.float32).reshape(-1, 4) for x in labels]
        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate((classes.reshape(-1, 1), bboxes), axis=1)
    return labels


def letterbox(image: np.array, label: np.array, new_size: Tuple[int]=(416,416)):
    """
    Args:
        image: input image for resizing with type of numpy array
        label: labels consting of classes and bboxes with type of numpy array
        new_size: a new size for resizing
    
    Returns:
        output: a resized image with padding
        label_out: adjusted labels with padding
    """
    assert new_size[0] == new_size[1], \
        f'The height {new_size[0]} and width {new_size[1]} should be equal'
    origin_h, origin_w, _ = image.shape

    ratio = max(origin_h / new_size[0], origin_w / new_size[0])
    resized_h, resized_w = round(origin_h / ratio), round(origin_w / ratio)
    assert max(resized_h, resized_w) != new_size[0], \
        f'The size {max(resized_h, resized_w)} is not {new_size[0]}'
    
    if resized_h < resized_w:
        left = right = 0
        diff = new_size[0] - resized_h
        if diff % 2 != 0:
            top = int(diff) // 2
            bottom = top + 1
        else:
            top = bottom = int(diff) // 2

    elif resized_h > resized_w:
        top = bottom = 0
        diff = new_size[0] - resized_w
        if diff % 2 != 0:
            left = int(diff) // 2
            right = left + 1
        else:
            left = right = int(diff) // 2

    else:
        output = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        return output, label

    # adjust label for padded image
    label_out = np.zeros_like(label)
    label_out[:, 0] = label[:, 0] # class number

    cx, cy = label[..., 1] * origin_w, label[..., 2] * origin_h
    w, h = label[..., 3] * origin_w, label[..., 4] * origin_h
    x1 = (cx - w / 2) / ratio + left
    x2 = (cx + w / 2) / ratio + right
    y1 = (cy - h / 2) / ratio + top
    y2 = (cy + h / 2) / ratio + bottom

    label_out[..., 1] = ((x2 + x1) / 2) / new_size[1]
    label_out[..., 2] = ((y2 + y1) / 2) / new_size[0]
    label_out[..., 3] = (x2 - x1) / new_size[1]
    label_out[..., 4] = (y2 - y1) / new_size[0]

    resized_img = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    output = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return output, label_out