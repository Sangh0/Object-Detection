import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        assert resnet_type in ('resnet50', 'resnet101'), \
            f'The {resnet_type} does not exist'
        if resnet_type is 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            self.backbone = models.resnet101(pretrained=pretrained)
        
        self.model = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )
        
        self.out_channels = 2048
        
    def forward(self, x):
        out = self.model(x)
        return out
    
def FrozenBN(model):
    for name, module in model.named_modules():
        if name == 'bn1':
            module.requires_grad = False
        if name in ('layer1', 'layer2', 'layer3', 'layer4'):
            for child_name, child_module in module.named_modules():
                if (len(child_name) > 1) and child_name[2:4] == 'bn':
                    child_module.requires_grad_ = False
    return model

def build_backbone():
    return FrozenBN(ResNet())