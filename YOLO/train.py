import os
import argparse
import time
import logging
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from .utils.dataset import ObstacleDetectionDataset
from .utils.scheduler import CosineWarmupLR
from .utils.callback import EarlyStopping, CheckPoint
from .test import evaluate
from .model.yolo import YOLOv3


class TrainModel(object):

    def __init__(
        self,
        model: nn.Module,
        iou_threshold: float,
        conf_threshold: float,
        nms_threshold: float,
        optimizer: str='sgd',
        epochs: int=300,
        weight_decay: float=1e-5,
        warmup_epochs: int=10,
        warmup_lr: float=0.01,
        end_lr: float=0.0001,
        lr_scheduling: bool=False,
        check_point: bool=True,
        early_stop: bool=False,
        weight_save_path: str='./weights',
        train_log_step: int=100,
    ):
        assert optimizer in ('sgd', 'adam')
        assert warmup_lr > end_lr

        self.logger = logging.getLogger('The logs of training model')
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
        self.logger.info(f'device is {self.device}...')

        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.model = model.to(self.device)

        self.epochs = epochs
        
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=warmup_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=warmup_lr,
                weight_decay=weight_decay,
            )

        self.lr_scheduling = lr_scheduling
        self.lr_scheduler = CosineWarmupLR(
            self.optimizer,
            epochs=epochs,
            lr_min=end_lr,
            warmup_epochs=warmup_epochs,
        )

        os.makedirs(weight_save_path, exist_ok=True)
        self.weight_save_dir = weight_save_dir
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)

        self.early_stop = early_stop
        self.es = EarlyStopping(patience=20, verbose=True, path=weight_save_path+'/early_stop.pt')

        self.writer = SummaryWriter()

        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step

    def fit(self, train_loader, valid_loader):
        self.logger.info('\nStart Training Model...!')
        start_train = time.time()
        for epoch in tqdm(range(self.epochs)):
            init_time = time.time()

            # training
            train_loss = self.train_on_batch(train_loader)

            # validating
            precision, recall, AP, f1, _, _, _ = evaluate(
                model=self.model,
                dataloader=valid_loader,
                iou_threshold=self.iou_thresh,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
            )

            self.logger.info(f'\n{"="*40} Epoch {epoch+1}/{self.epochs} {"="*40}'
                             f'\n{" "*10}time: {time.time()-init_time:.3f}s'
                             f'  lr = {self.optimizer.param_groups[0]["lr"]}')
            self.logger.info(f'train loss: {train_loss:.3f}'
                             f'valid precision: {precision.mean():.3f}, recall: {recall.mean():.3f}, mAP: {AP.mean():.3f}')

            for idx, yolo_layer in enumerate(self.model.yolo_layers):
                self.writer.add_scalar(f'Train/box_loss_{idx+1}', yolo_layer.metrics['box_loss'], epoch)
                self.writer.add_scalar(f'Train/conf_loss_{idx+1}', yolo_layer.metrics['conf_loss'], epoch)
                self.writer.add_scalar(f'Train/cls_loss_{idx+1}', yolo_layer.metrics['cls_loss'], epoch)
            self.writer.add_scalar('Train/total_loss', train_loss.item(), epoch)

            self.writer.add_scalar('Valid/precision', precision.mean(), epoch)
            self.writer.add_scalar('Valid/recall', recall.mean(), epoch)
            self.writer.add_scalar('Valid/mAP', AP.mean(), epoch)
            self.writer.add_scalar('Valid/f1_score', f1.mean(), epoch)

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch)

            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = self.weight_save_dir+f'/check_point_{epoch+1}.pt'
                self.cp(1-AP, self.model, path)

            if self.early_stop:
                self.es(1-AP, self.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        self.writer.close()
        self.logger.info(f'\nTotal time for training is {time.time()-start_train:.3f}s')


    def train_on_batch(self, train_loader):
        self.model.train()

        for batch, (images, labels, _) in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            _, loss = self.model(images, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batch+1) % self.train_log_step == 0:
                self.logger.info(f'\n{" "*20} Train Batch {batch+1}/{len(train_loader)} {" "*20}'
                                 f'\ntrain loss: {loss:.3f}')
            
        return loss.detach().cpu().item()


def get_args_parser():
    parser = argparse.ArgumentParser(description='YOLOv3 Training', add_help=False)

    # training parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='The epoch number for training model')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay for regularization')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='The epoch number for warmup scheduler')
    parser.add_argument('--warmup_lr', type=float, default=1e-2,
                        help='The learning rate for warmup scheduler')
    parser.add_argument('--end_lr', type=float, default=1e-7,
                        help='The final learning rate for warmup scheduler')
    parser.add_argument('--lr_scheduling', action='store_true'
                        help='whether to apply cosine with linear warmup scheduler')
    parser.add_argument('--check_point', action='store_true'
                        help='save a weight of model')
    parser.add_argument('--early_stop', action='store_true',
                        help='save a weight in last epoch if training is stopping')
    parser.add_argument('--weights_save_path', type=str, default='./weights',
                        help='a directory for saving weights')
    parser.add_argument('--train_log_step', type=int, default=100,
                        help='print the logs in training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='optimizer')

    # dataset directory
    parser.add_argument('--data_yaml_path', type=str, required=True,
                        help='a file name of dataset yaml format')
    
    # augmentation hyperparameters
    parser.add_argument('--horizontal_p', type=float, default=0.5,
                        help='a probability for horizontal flipping')
    parser.add_argument('--rotate_p', type=float, default=0.5,
                        help='a probability for rotation')
    parser.add_argument('--clahe_p', type=float, default=0.5,
                        help='a probability for clahe')
    parser.add_argument('--brightness_p', type=float, default=0.5,
                        help='a probability for brightness in color jitter')
    parser.add_argument('--contrast_p', type=float, default=0.5,
                        help='a probability for contrast in color jitter')
    parser.add_argument('--saturation_p', type=float, default=0.5,
                        help='a probability for saturation in color jitter')
    
    # loss function parameters
    parser.add_argument('--obj_pw', type=float, default=1.,
                        help='positive weight in objectness loss')
    parser.add_argument('--cls_pw', type=float, default=1.,
                        help='positive weight in classification loss')
    parser.add_argument('--label_smooth', type=float, default=0.,
                        help='label smoothing in loss function')

    # threshold hyperparameters
    parser.add_argument('--iou_threshold', type=float, default=0.2,
                        help='a threshold value of iou score')
    parser.add_argument('--conf_threshold', type=float, default=0.001,
                        help='a threshold value of confidence score')
    parser.add_argument('--nms_threshold', type=float, default=0.2,
                        help='a threshold value of non maximum suppersion')
    
    return parser


def main(args):

    train_data = ObstacleDetectionDataset(
        path=args.path,
        subset='train',
        img_size=(416, 416),
        transforms_=True,
        horizontal_p=args.horizontal_p,
        rotate_p=args.rotate_p,
        clahe_p=args.clahe_p,
        brightness_p=args.brightness_p,
        contrast_p=args.contrast_p,
        saturation_p=args.saturation_p,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_data = ObstacleDetectionDataset(
        path=args.path,
        subset='valid',
        img_size=(416, 416),
        transforms_=False,
        horizontal_p=None,
        rotate_p=None,
        clahe_p=None,
        brightness_p=None,
        contrast_p=None,
        saturation_p=None,
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    yolov3 = YOLOv3(obj_pw=args.obj_pw, cls_pw=args.cls_pw, label_smooth=args.label_smooth)
    summary(model, (3, 416, 416), device='cpu')

    model = TrainModel(
        model=yolov3,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        optimizer=args.optimizer,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        end_lr=args.end_lr,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
        weight_save_path=args.weight_save_path,
        train_log_step=args.train_log_step,
    )

    model.fit(train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)