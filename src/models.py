import torch
from torch import nn
from efficientnet_pytorch.model import EfficientNet

# Reference: https://www.kaggle.com/hmendonca/efficientnet-pytorch
model_default_weight = {
    'efficientnet-b0': './enet_weight/efficientnet-b0-08094119.pth',
    'efficientnet-b1': './enet_weight/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': './enet_weight/efficientnet-b2-27687264.pth',
    'efficientnet-b3': './enet_weight/efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': './enet_weight/efficientnet-b4-e116e8b3.pth',
}

class enet(nn.Module):
    def __init__(self, model_type='efficientnet-b0', pretrained=True, out_dim=5):
        super(enet, self).__init__()
        self.base = EfficientNet.from_name(model_type)
        if pretrained:
            # self.base = EfficientNet.from_pretrained(model_type)
            self.base.load_state_dict(torch.load(model_default_weight[model_type]))
        self.myfc = nn.Linear(self.base._fc.in_features, out_dim)
        self.base._fc = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        out = self.myfc(x)
        return out