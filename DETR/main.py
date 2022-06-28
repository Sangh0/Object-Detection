import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Set DETR training', add_help=False)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Name of the CNN backbone to use')
    parser.add_argument('--pretrained', default=True, type=bool,
                        help='Load pre-trained weights of backbone network')
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='The dimension of embedding sequence')
    parser.add_argument('--n_head', default=8, type=int,
                        help='Number of multi head attention block')
    parser.add_argument('--feedforward_dim', default=2048, type=int,
                        help='The dimension of feedforward layers in the transformer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate of the transformer')
    parser.add_argument('--encoder_layers', default=6, type=int,
                        help='Number of encoding layers in the transformer')
    parser.add_argument('--decoder_layers', default=6, type=int,
                        help='Number of decoding layer sin the transformer')
    parser.add_argument('--num_classes', default=80, type=int,
                        help='Number of total classes')
    parser.add_argument('--num_queries', default=12, type=int,
                        help='Maximum number of objects in an image')
    parser.add_argument('--pos_embed_dim', default=512, type=int,
                        help='The dimension of position encoding vector')
    parser.add_argument('--EPOCH', default=10000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--early_stop_patience', default=100, type=int)
    parser.add_argument('--img_height', default=832, type=int,
                        help='Size of height of input image')
    parser.add_argument('--img_width', default=832, type=int,
                        help='Size of width of input image')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='Number of CPU cores')
    return parser

def main(args):
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    from dataset import ObjectDetectionDataset
    from models.detr import DETR
    from models.matcher import (
        HungarianMatcher, 
        box_cxcywh_to_xyxy, 
        box_iou, 
        generalized_box_iou
    )
    from torchscheduler import PolynomialLRDecay
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from torchcallback import CheckPoint, EarlyStopping
    from train import train_step

    device = torch.device('cuda')

    data_path = '#####'

    train_loader = DataLoader(
        ObjectDetectionDataset(path=data_path, subset='train', height=args.height, width=args.width),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        ObjectDetectionDataset(path=data_path, subset='valid', height=args.height, width=args.width),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DETR(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        backbone_type=args.backbone,
        pretrained=args.pretrained,
        pos_embed_dim=args.embed_dim,
    )

    param_dicts = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {
            'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
            'lr': args.lr_backbone
        },
    ]

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.EPOCH)
    matcher = HungarianMatcher()
    metric_map = MeanAveragePrecision(box_format='cxcywh',
                                      iou_type='bbox',
                                      iou_thresholds=0.5)

    checkpoint = CheckPoint(verbose=True)
    early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True)

    history = train_step(
        model=model,
        train_data=train_loader,
        validation_data=valid_loader,
        epochs=args.EPOCH,
        matcher=matcher,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        learning_rate_scheduler=True,
        check_point=True,
        early_stop=False,
    )

if __name__ == '__main__':
    main(get_args_parser)