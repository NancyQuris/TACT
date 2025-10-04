from copy import deepcopy
import torch.nn as nn
import torchvision.models as models
from .identity import Identity

PRETRAINED_WEIGHT_DICT = {
    'vit_b_32': models.ViT_B_32_Weights.IMAGENET1K_V1,
    'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1,
}

N_OUTPUT = {
    'vit_b_32': 768, 
    'vit_b_16': 768,
}


class Network(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # load model 
        network_class = getattr(models, model_name)
        # by default, model pretrained on imagenet is loaded
        model = network_class(weights=PRETRAINED_WEIGHT_DICT[model_name])
        
        self.classifier = deepcopy(model.heads)
        self.classifier.weight = self.classifier.head.weight
        model.heads = Identity()
        self.featurizer = model 
        
        self.n_output = N_OUTPUT[model_name]
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def forward(self, x):
        return self.network(x)