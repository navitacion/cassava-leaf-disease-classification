from torch import nn
import torch
# Pytorch Image Model
# https://rwightman.github.io/pytorch-image-models/
import timm
from timm import create_model


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        # Pointwise convolution
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x

        # B,C,W,H　→　B,C,N
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2] * X.shape[3])
        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2] * X.shape[3])

        # Transpose
        proj_query = proj_query.permute(0, 2, 1)

        S = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(S)

        # Transpose
        attention_map = attention_map_T.permute(0, 2, 1)

        # Self-Attention Map
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        out = x + self.gamma * o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])

        return out, attention_map


class Timm_model(nn.Module):
    def __init__(self, model_name, pretrained=True, out_dim=5):
        super(Timm_model, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained)

        if 'efficientnet' in model_name:
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=out_dim)
        elif 'vit' in model_name:
            self.base.head = nn.Linear(in_features=self.base.head.in_features, out_features=out_dim)
        else:
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=out_dim)

    def forward(self, x):
        return self.base(x)


class Ensembler(nn.Module):
    def __init__(self, model_names, weights, pretrained=True, out_dim=5):
        super(Ensembler, self).__init__()
        self.models = nn.ModuleList([Timm_model(model_name, pretrained, out_dim) for model_name in model_names])
        self.weights = weights

    def forward(self, x):
        outs = []
        for m in self.models:
            outs.append(m(x))

        out = torch.zeros_like(outs[0])
        for o, w in zip(outs, self.weights):
            out += o * w

        del outs
        return out


if __name__ == '__main__':
    # Print Timm Models
    model_names = timm.list_models(pretrained=True)
    print(model_names)

    # models = ['resnet18', 'efficientnet_b0']
    # weights = [0.4, 0.6]
    # net = Ensembler(models, weights)
    #
    # z = torch.randn(3, 3, 512, 512)
    # out = net(z)
    #
    # print(out.size())
