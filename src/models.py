from torch import nn
from efficientnet_pytorch.model import EfficientNet


class enet(nn.Module):
    def __init__(self, model_type='efficientnet-b0', pretrained=True, out_dim=5):
        super(enet, self).__init__()
        self.base = EfficientNet.from_name(model_type)
        if pretrained:
            self.base = EfficientNet.from_pretrained(model_type)
        self.myfc = nn.Linear(self.base._fc.in_features, out_dim)
        self.base._fc = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        out = self.myfc(x)
        return out