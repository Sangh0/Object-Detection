"""
path : dataset/
├── train
│    ├─ images
│       ├─ img1.jpg
│       ├─ ...
│    ├─ annotations
│       ├─ anno1.txt 
│       ├─ ...
├── valid
│    ├─ images
│       ├─ img100.jpg
│       ├─ ...
│    ├─ annotations
│       ├─ anno100.txt
│       ├─ ... 
├── test
│    ├─ images
│       ├─ img1000.jpg
│       ├─ ...
│    ├─ annotations
│       ├─ anno1000.txt 
│       ├─ ...
"""
import cv2
from glob import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A

width = 1200
height = 800

class ObjectDetectionDataset(Dataset):
    def __init__(self, path, subset, img_size, transforms_=None):
        super().__init__()
        assert subset in ('train', 'valid', 'test'), \
            f'unknown {subset} set'
        self.image_files = sorted(glob(path+'/image/'+subset+'/*.jpg'))
        self.anno_list = [
            open(file, 'r') 
            for file in sorted(glob(path+'/annotation/'+subset+'/*.txt'))
        ]
        assert len(self.image_files) == len(self.anno_list), \
            f'The size of {self.image_files} and {self.anno_list} is different.'
        
        self.subset = subset
        self.img_size = img_size
        self.transforms_ = transforms_
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        """
        Explain Parameters of obtions of albumentations augmentation 
            # format: str
                - 1. pascal_voc : (xmin, ymin, xmax, ymax) with integer coordinates
                - 2. albumentations : (xmin, ymin, xmax, ymax) with normalized coordinates
                - 3. coco : (xmin, ymin, xmax, ymax) with integer coordinates
                - 4. yolo : (xcenter, ycenter, width, height) with normalized coordinates
        
            # min_area: integer in [0,255]
                - If the area of the box becomes smaller than min_area after augmentation, the box is delted
        
            # min_visibility: float in [0,1]
                - If the ratio of the box becomes smaller than min_visibility after augmentation, the box is delted 
        
            # label_fiels:
                - 'labels', 'class_labels', 'class_categories', etc.
        """
        
        self.train_augment = A.Compose([
            A.HorizontalFlip(),
            A.Resize(height=height, width=width, p=1),
        ],
            bbox_params=A.BboxParams(
                format='yolo',  
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        )
        self.valid_augment = A.Compose([
            A.Resize(height=height, width=width, p=1),
        ],
            bbox_params=A.BboxParams(
                format='yolo',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        )

    def __getitem__(self, idx):
        images = self.get_image(idx)
        bboxes, labels = self.get_label(idx)
        if self.subset is 'train':
            if self.transforms_ is not None:
                augmented = self.train_augment(image=images, bboxes=bboxes, class_labels=labels)
                images, bboxes, labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
        elif self.subset is 'valid':
            if self.transforms_ is not None:
                augmented = self.train_augment(image=images, bboxes=bboxes, class_labels=labels)
                images, bboxes, labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
        return self.totensor(images), bboxes, labels
    
    def get_image(self, row):
        images = cv2.imread(self.image_files[row], cv2.IMREAD_COLOR)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        return images
    
    def get_label(self, row):
        annos = self.anno_list[row].readlines()
        bbox_list, label_list = [], []
        # create bounding box list
        for anno in annos:
            sub_bbox_list = []
            for i, anno_sub in enumerate(anno.rstrip().split(' ')):
                if i==0:
                    label_list.append(int(anno_sub))
                else:
                    sub_bbox_list.append(float(anno_sub))
            bbox_list.append(sub_bbox_list)
        return torch.FloatTensor([bbox_list]), torch.LongTensor([label_list])#bbox_list, label_list
        
    def center_to_corner(self, coordinate):
        # (cx,cy,w,h) -> (x1,y1,x2,y2)
        x1y1 = coordinate[...,:2] - (coordinate[...,2:4])/2
        x2y2 = coordinate[...,:2] + (coordinate[...,2:4])/2
        corner_boxes = torch.cat([x1y1, x2y2], dim=-1)
        return corner_boxes
    
    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    path = '###'
    batch_size = 1
    
    train_loader = DataLoader(
        ObjectDetectionDataset(path=path, subset='train', img_size=height),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        ObjectDetectionDataset(path=path, subset='valid', img_size=height),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )