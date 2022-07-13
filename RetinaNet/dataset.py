from PIL import Image
from glob import glob

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    def __init__(
        self,
        path,
        height=None,
        width=None,
        transforms_=None,
    ):
        
        self.anno_files = sorted(glob(path + 'annotations/*.txt'))
        self.image_files = [path + 'images/' + anno.split('/')[-1].split('.')[0] + '.jpg' for anno in self.anno_files]
        assert len(self.anno_files) == len(self.image_files), \
            f'annotations size {len(self.anno_files)} and image size {len(self.image_files)} is different'
        
        self.box_list, self.label_list = [], []
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as f:
                sub_bbox_list, sub_label_list = [], []
                for anno in f.readlines():
                    sub2_bbox_list = []
                    for i, sub_anno in enumerate(anno.rstrip().split(' ')):
                        sub_label_list.append(int(sub_anno)) if i==0 \
                            else sub2_bbox_list.append(float(sub_anno))
                    sub_bbox_list.append(sub2_bbox_list)
                self.box_list.append(sub_bbox_list)
                self.label_list.append(sub_label_list)
        
        self.height = height
        self.width = width
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.anno_files)
    
    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx]).convert('RGB')
        labels = self.label_list[idx] 
        bboxes = self.box_list[idx]
        if self.transforms_ is not None:
            images, labels, bboxes = self.transforms_(images, labels, bboxes)
        return images, labels, bboxes