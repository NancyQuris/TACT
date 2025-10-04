from copy import deepcopy
import torch.nn as nn 
from torchvision.models import vit_b_16

from .identity import Identity

class Network(nn.Module):
    def __init__(self, num_classes, pretrained, group_num=None, weights=None, **kwargs):
        super(Network, self).__init__()
        vit = vit_b_16(pretrained=pretrained, **kwargs)    
        del vit.heads
        vit.heads = Identity()
        
        self.featurizer = vit
        # replace normalization layer
        if group_num != None:  
            for name, module in self.featurizer.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn = getattr(self.featurizer, name)
                    gn = nn.GroupNorm(group_num, bn.num_features)
                    nn.init.constant_(gn.weight, 1)
                    nn.init.constant_(gn.bias, 0)
                    setattr(self.featurizer, name, gn)
        
        self.classifier = nn.Linear(768, num_classes)
        
        self.network = nn.Sequential(
            self.featurizer,
            self.classifier
        )
        self.n_output = 768

        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        return self.network(x)
    
    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))
