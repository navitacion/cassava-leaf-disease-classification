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


class EFNet_b0_with_Attention(nn.Module):
    def __init__(self, pretrained=True):
        super(EFNet_b0_with_Attention, self).__init__()
        self.timm = Timm_model('efficientnet_b0', pretrained=pretrained)

        self.attemtion_1 = Self_Attention(in_dim=24)
        # self.attemtion_2 = Self_Attention(in_dim=112)

    def forward(self, x):
        out = self.timm.base.conv_stem(x)
        out = self.timm.base.bn1(out)
        out = self.timm.base.act1(out)
        out = self.timm.base.blocks[0](out)
        out = self.timm.base.blocks[1](out)
        out, _ = self.attemtion_1(out)
        out = self.timm.base.blocks[2](out)
        out = self.timm.base.blocks[3](out)
        out = self.timm.base.blocks[4](out)
        # out, _ = self.attemtion_2(out)
        out = self.timm.base.blocks[5](out)
        out = self.timm.base.blocks[6](out)

        out = self.timm.base.conv_head(out)
        out = self.timm.base.bn2(out)
        out = self.timm.base.act2(out)
        out = self.timm.base.global_pool(out)
        out = self.timm.base.classifier(out)

        return out




if __name__ == '__main__':
    # Print Timm Models
    model_names = timm.list_models(pretrained=True)
    # print(model_names)

    net = Timm_model(model_name='efficientnet_b0', pretrained=False)
    # print(net)
    inp = torch.randn(4, 3, 224, 224)

    out, out2 = net(inp)

    print(out.size())
    print(out2.size())