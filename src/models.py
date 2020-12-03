from torch import nn

# Pytorch Image Model
# https://rwightman.github.io/pytorch-image-models/
import timm
from timm import create_model

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



if __name__ == '__main__':
    # Print Timm Models
    model_names = timm.list_models(pretrained=True)
    print(model_names)