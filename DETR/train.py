import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchscheduler import PolynomialLRDecay
from torchcallback import CheckPoint, EarlyStopping
from models.detr import DETR
from models.matcher import *
from dataset import ObjectDetectionDataset

device = torch.device('cuda')

lr = 5e-2
EPOCH = 10000
num_classes = 80 # dataset의 클래스 갯수
num_queries = 12 # 한 이미지에 들어있는 최대 object의 수

model = DETR(
    num_classes=num_classes,
    num_queries=num_queries,
    backbone_type='resnet50',
    pretrained=True,
    pos_embed_dim=512,
)
param_dicts = [
    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]},
    {
        'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
        'lr': 1e-5
    },
]

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=EPOCH)
matcher = HungarianMatcher()

es_save_path = './weights/earlystop'
cp_save_path = './weights/checkpoint'
metric_map = MeanAveragePrecision(box_format='cxcywh', 
                                  iou_type='bbox',
                                  iou_thresholds=0.5)
checkpoint = CheckPoint(verbose=True, path=cp_save_path)
early_stopping = EarlyStopping(patience=100, verbose=True, path=es_save_path)

def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

logger = SummaryWriter(comment=experiment)

def validate_on_batch(model, 
                      validation_data,
                      matcher,
                      num_classes,
                      num_queries):
    model.eval()
    with torch.no_grad():
        vbatch_loss, vbatch_giou, vbatch_map = 0, 0, 0
        for vbatch, (images, targets) in enumerate(validation_data):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            
            indices = matcher(outputs, targets)
            target_classes_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
            target_classes = torch.full(outputs['pred_logits'].shape[:2], num_classes=num_classes,
                                        dtype=torch.int64, device=device)
            target_classes[idx] = target_classes_o
            # calculate classification loss
            val_loss_ce = F.cross_entropy(outputs['pred_logits'].transpose(1,2), target_classes)
            # calculate bounding box loss
            idx = _get_src_permutation_idx(indices)
            src_bboxes = outputs['pred_bboxes'][idx]
            target_bboxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_bbox = F.l1_loss(src_bboxes, target_bboxes, reduction='none')
            val_loss_bbox = loss_bbox.sum() / num_queries
            # calculate giou loss
            giou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_bboxes),
                box_cxcywh_to_xyxy(target_bboxes)))
            val_loss_giou = (1-giou).sum() / num_queries
            # calculate mean AP
            val_map = metric_map(outputs, targets)
            
            val_losses = val_loss_ce + val_loss_bbox + val_loss_giou
            vbatch_loss += val_losses.item()
            vbatch_giou += giou.item()
            vbatch_map += val_map.item()
    
    return vbatch_loss/(vbatch+1), vbatch_giou/(vbatch+1), vbatch_map/(vbatch+1)
    
def train_on_batch(model,
                   train_data,
                   matcher,
                   num_classes,
                   num_queries):
    batch_loss, batch_giou, batch_map = 0, 0, 0
    for batch, (images, targets) in enumerate(train_data):
        model.train()
        optimizer.zero_grad()
        
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        
        indices = matcher(outputs, targets)
        target_classes_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(outputs['pred_logits'].shape[:2], num_classes=num_classes, 
                                    dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o
        # calculate classification loss
        loss_ce = F.cross_entropy(outputs['pred_logits'].transpose(1,2), target_classes)
        # calculate bounding box loss
        idx = _get_src_permutation_idx(indices)
        src_bboxes = outputs['pred_bboxes'][idx]
        target_bboxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_bboxes, target_bboxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_queries
        # calculate giou loss
        giou = torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_bboxes),
            box_cxcywh_to_xyxy(target_bboxes)))
        loss_giou = (1-giou).sum() / num_queries
        # calculate mean AP
        map_score = metric_map(outputs, targets)
        
        losses = loss_ce + loss_bbox + loss_giou
        batch_loss += losses.item()
        batch_giou += giou.item()
        batch_map += map_score.item()
    
    return losses/(batch+1), batch_giou/(batch+1), batch_map/(batch+1)

def train_step(
    model,
    train_data,
    validation_data,
    epochs,
    matcher,
    num_classes,
    num_queries,
    learning_rate_scheduler=False,
    check_point=False,
    early_stop=False,
    last_epoch_save_path='./model/last_checkpoint.pt'
):
    
    loss_list, giou_list, map_list = [], [], []
    val_loss_list, val_giou_list, val_map_list = [], [], []
    
    print('Start Model Training!')
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        init_time = time.time()
        
        # train step
        train_loss, train_giou, train_map = train_on_batch(
            model, train_data, matcher, num_classes, num_queries
        )
        loss_list.append(train_loss)
        giou_list.append(train_giou)
        map_list.append(train_giou)
        
        # validate step
        val_loss, val_giou, val_map = validate_on_batch(
            model, validation_data, matcher, num_classes, num_queries
        )
        val_loss_list.append(val_loss)
        val_giou_list.append(val_giou)
        val_map_list.append(val_map)

        logger.add_scalar('loss', {'train_loss':train_loss, 'val_loss':val_loss}, epoch+1)
        logger.add_scalar('giou', {'train_giou':train_giou, 'val_giou':val_giou}, epoch+1)
        logger.add_scalar('map', {'train_map':train_map, 'val_map':val_map}, epoch+1)
        
        print(f'\n[Epoch {epoch+1}/{epochs}]'
              f'  [time: {time.time()-init_time:.3f}s]'
              f'  [lr = {optimizer.param_groups[0]["lr"]}]')
        print(f'training log:'
              f'\n[train loss: {train_loss:.3f}]'
              f'  [train GIoU: {train_giou:.3f}]'
              f'  [train mean AP: {train_map:.3f}]')
        print(f'validating log:'
              f'\n[valid loss: {val_loss:.3f}]'
              f'  [valid GIoU: {val_giou:.3f}]'
              f'  [valid mean AP: {val_map:.3f}]')
        
        if learning_rate_scheduler:
            lr_scheduler.step()
            
        if check_point:
            checkpoint(val_loss, model, cp_save_path+f'{epoch+1}.pt')
        
        if early_stop:
            assert check_point==False, \
                'Choose between Early Stopping and Check Point'
            early_stopping(val_loss, model, es_save_path+f'{epoch+1}.pt')
            if early_stopping.early_stop:
                print('\n##########################\n'
                      '##### Early Stopping #####\n'
                      '##########################')
                break
        
    if early_stop==False and check_point==False:
        torch.save(model.state_dict(), last_epoch_save_path)
        print('Saving model of last epoch.')
        
    print(f'\nTotal tiem for training is {time.time()-start_time:.3f}s')
    
    # end logging of tensorboard
    logger.close()

    """
    # check tensorboard on real-time
    %reload_ext tensorboard
    %tensorboard --logdir lightning_logs/
    """

    return {
        'model': model,
        'loss': loss_list,
        'giou': giou_list,
        'map': map_list,
        'val_loss': val_loss_list,
        'val_giou': val_giou_list,
        'val_map': val_map_list,
    }