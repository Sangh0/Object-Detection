import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from .utils.dataset import ObstacleDetectionDataset
from .model.yolo import YOLOv3
from .utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class


def evaluate(model, dataloader, iou_threshold, conf_threshold, nms_threshold, device):
    model.eval()

    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for _, images, targets in tqdm(dataloader):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return precision, recall, AP, f1, ap_class, inference_time, 