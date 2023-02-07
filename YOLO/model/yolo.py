from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBlock, DarkNetBackbone
from ..utils.loss import MSELoss, BCEWithLogitsLoss
from ..utils.util import build_taregts
from ..config import LAST_LAYER_DIM, NUM_ANCHORS_PER_SCALE, NUM_ATTRIB, ANCHORS, NUM_CLASSES, IGNORE_THRESH, NOOBJ_COEFF, EPSILON


class DetectionBlock(nn.Module):
    """
    Detection Block
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(DetectionBlock, self).__init__()
        assert out_dim % 2 == 0, f'The output dimension {out_dim} is not even'
        hidden_dim = out_dim // 2

        self.block1 = ConvBlock(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.block2 = ConvBlock(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.block3 = ConvBlock(out_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.block4 = ConvBlock(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.block5 = ConvBlock(out_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.block6 = ConvBlock(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(out_dim, LAST_LAYER_DIM, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        branch = self.block5(x)
        x = self.block6(branch)
        x = self.conv_out(x)
        return x, branch


class Upsample(nn.Module):
    """
    Upsampling Layer to detect smaller objects in feature map
    """
    def __init__(
        self, 
        scale_factor: float=2., 
        mode: str='nearest',
    ):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor):
        out = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
        )
        return out


class YOLOLayer(nn.Module):

    def __init__(
        self, 
        scale: str, 
        stride: int, 
        obj_pw: float, 
        cls_pw: float,
        label_smooth: float,
    ):
        super(YOLOLayer, self).__init__()
        assert scale in ('s', 'm', 'l'), 'you should select between s, m and l'
        
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        else:
            idx = (6, 7, 8)

        self.anchors = torch.Tensor([ANCHORS[i] for i in idx])
        self.stride = stride

        self.mse_loss = MSELoss()
        self.bce_loss = BCELoss()
        
    def forward(self, x, taregts):
        batch_size = x.size(0)
        grid_size = x.size(2)

        prediction = x.view(
            batch_size, NUM_ANCHORS_PER_SCALE, NUM_ATTRIB, grid_size, grid_size
        ).permute(0, 1, 3, 4, 2)

        # calculate offsets for each grid
        self.anchors = self.anchors.to(x.device).float()
        grid_tensor = torch.arange(grid_size, dtype=torch.float, device=x.device).repeat(grid_size, 1)
        grid_x = grid_tensor.view([1, 1, grid_size, grid_size])
        grid_y = grid_tensor.t().view([1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors], dtype=torch.float, device=x.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, -1, 1, 1))

        # get outputs
        pred_cx = torch.sigmoid(prediction[..., 0]) + grid_x
        pred_cy = torch.sigmoid(prediction[..., 1]) + grid_y
        pred_w = torch.exp(prediction[..., 2]) * anchor_w
        pred_h = torch.exp(prediction[..., 3]) * anchor_h
        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=4).view((batch_size, -1, 4)) * self.stride

        pred_conf = torch.sigmoid(prediction[..., 4]).view(batch_size, -1, 1)
        pred_cls = torch.sigmoid(prediction[..., 5:]).view(batch_size, -1, NUM_CLASSES)

        out = torch.cat([pred_bbox, pred_conf, pred_cls], dim=1)

        if targets is None:
            return out

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_taregts(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            targets=taregts,
            anchors=scaled_anchors,
            ignore_thresh=IGNORE_THRESH,
        )
        
        # calculate losses
        loss_x = self.mse_loss(pred_cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(pred_cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(pred_w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(pred_h[obj_mask], th[obj_mask])
        box_loss = loss_x + loss_y + loss_w + loss_h

        obj_conf_loss = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        noobj_conf_loss = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        conf_loss = NOOBJ_COEFF * noobj_conf_loss + obj_conf_loss

        cls_loss = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        total_loss = box_loss + conf_loss + cls_loss

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        cls_acc = 100 * class_mask[obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + EPSILON)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + EPSILON)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + EPSILON)


        self.metrics = {
            'box_loss': box_loss.detach().cpu().item(),
            'conf_loss': conf_loss.detach().cpu().item(),
            'cls_loss': cls_loss.detach().cpu().item(),
            'cls_acc': cls_acc.detach().cpu().item(),
            'precision': precision.detach().cpu().item(),
            'recall50': recall50.detach().cpu().item(),
            'recall75': recall75.detach().cpu().item(),
        }

        return out, total_loss


class YOLOv3(nn.Module):
    """
    YOLOv3 Network
    """
    def __init__(
        self, 
        obj_pw: float,
        cls_pw: float,
        label_smooth: float,
        in_dim: int=3, 
        num_filters: int=32, 
        repeat_list: List[int]=[1,2,8,8,4],
    ):
        super(YOLOv3, self).__init__()
        self.backbone = DarkNetBackbone(
            in_dim=in_dim, 
            num_filters=num_filters, 
            repeat_list=repeat_list, 
            task='detection',
        )
        
        self.detect1 = DetectionBlock(in_dim=1024, out_dim=1024) # 13 x 13 x 1024, stride: 416 / 13
        self.conv1 = ConvBlock(in_dim=512, out_dim=256, kernel_size=1, stride=1, padding=0)  # 13 x 13 x 512 to 13 x 13 x 256
        self.upsample1 = Upsample(scale_factor=2) # 13 x 13 x 256 to 26 x 26 x 256
        self.yololayer1 = YOLOLayer(scale='l', stride=32, obj_pw=obj_pw, cls_pw=cls_pw, label_smooth=label_smooth)

        self.detect2 = DetectionBlock(in_dim=768, out_dim=512) # 26 x 26 x 512, stride: 416 / 26
        self.conv2 = ConvBlock(in_dim=256, out_dim=128, kernel_size=1, stride=1, padding=0)  # 26 x 26 x 256 to 26 x 26 x 128
        self.upsample2 = Upsample(scale_factor=2) # 26 x 26 x 128 to 52 x 52 x 128
        self.yololayer2 = YOLOLayer(scale='m', stride=16, obj_pw=obj_pw, cls_pw=cls_pw, label_smooth=label_smooth)

        self.detect3 = DetectionBlock(in_dim=384, out_dim=256)
        self.yololayer3 = YOLOLayer(scale='s', stride=8, obj_pw=obj_pw, cls_pw=cls_pw, label_smooth=label_smooth)

        self.yolo_layers = [self.yololayer1, self.yololayer2, self.yololayer3]

        self._init_weight_()

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor]=None):
        loss = 0

        tmp1, tmp2, tmp3 = self.backbone(x)

        # 1024 x 13 x 13
        out1, branch1 = self.detect1(tmp1)
        out1, loss1 = self.yololayer1(out1)
        loss += loss1

        # 512 x 26 x 26
        branch1 = self.conv1(branch1)
        branch1 = self.upsample1(branch1)
        tmp2 = torch.cat((branch1, tmp2), dim=1)
        out2, branch2 = self.detect2(tmp2)
        out2, loss2 = self.yololayer2(out2)
        loss += loss2

        # 256 x 52 x 52
        branch2 = self.conv2(branch2)
        branch2 = self.upsample2(branch2)
        tmp3 = torch.cat((branch2, tmp3), dim=1)
        out3, _ = self.detect3(tmp3)
        out3, loss3 = self.yololayer3(out3)
        loss += loss3

        out = torch.cat([out1, out2, out3], dim=-1)
        
        return out if targets is None else out, loss

    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)