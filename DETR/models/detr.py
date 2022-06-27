import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import build_backbone, ResNet
from positionencoding import PositionEncoding
from transformer import Transformer

backbone = build_backbone()

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) \
            for n, k in zip([in_dim] + h, h + [out_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    """
    params:
        - num_classes: number of object classes.
        - num_queries: number of object queries. We set maximum number of objects.
        - backbone_type: type of CNN backbone, Ex) resnet50, resnet101
        - frozen_bn: Frozen Batch Normalization.
        - pretrained: get pre-trained weightes of backbone.
        - pos_emb_dim: dimension of embedding vector.
    
    returns:
        - pred_logits: the class for all queries, 
                       represented as [batch_size x num_queires x (num_class+1)].
        - pred_bboxes: the bounding boxes coordinates for all queries with normalized,
                       represented as [center_x, center_y, height, width] in [0,1].
        - aux_output: the output for calculating auxiliary loss.
    """
    def __init__(
        self,
        num_classes,
        num_queries=100,
        backbone_type='resnet50',
        frozen_bn=None,
        pretrained=True,
        pos_emb_dim=512,
    ):
        super(DETR, self).__init__()
        self.backbone = backbone if frozen_bn is not None else \
            ResNet(resnet_type=backbone_type, pretrained=pretrained)
        self.position_encoding = PositionEncoding(n_dim=pos_emb_dim)
        self.transformer = Transformer()
        hidden_dim = self.transformer.embed_dim
        backbone_out_channels = self.backbone.out_channels
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(  # self.conv
            backbone_out_channels, hidden_dim, kernel_size=1
        )
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
    def forward(self, x):
        features = self.backbone(x)
        projection = self.input_proj(features)
        pos_embed = self.position_encoding(features)
        
        """ Check each shape of parameters in transformer
            Let input shape is (1,512,7,7)
            - projection shape: (49, 1, 512)
            - pos_embed shape: (49, 1, 512)
            - query_embed.weight shape: (num_queries, hidden_dim)
        """
        hs = self.transformer(projection, self.query_embed.weight, pos_embed)[0]
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class, 
            'pred_bboxes': outputs_coord,
        }