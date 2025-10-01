
from copy import deepcopy
import torch.nn as nn 

from transformers import DistilBertModel

class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


class Network(nn.Module):
    ''' the input to forward and featurizer should be the transformer input'''
    def __init__(self, num_classes, pretrained=True, group_num=None, weights=None, low_dim_proj=False):
        super(Network, self).__init__()
        self.featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")

        if group_num != None:  
            for name, module in self.featurizer.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn = getattr(self.featurizer, name)
                    gn = nn.GroupNorm(group_num, bn.num_features)
                    nn.init.constant_(gn.weight, 1)
                    nn.init.constant_(gn.bias, 0)
                    setattr(self.featurizer, name, gn)

        self.n_output = self.featurizer.config.hidden_size
        self.pre_classifier = nn.Linear(self.n_output, self.n_output) 
        if low_dim_proj:
            self.classifier = nn.Sequential(
                nn.Linear(self.n_output, 2),
                nn.Linear(2, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.n_output, num_classes)
        
        self.network = nn.Sequential(self.featurizer, 
                                     self.pre_classifier,
                                     nn.ReLU(),
                                     self.classifier)
        
        if weights is not None:
            self.load_state_dict(deepcopy(weights))
        
    def forward(self, x):
        return self.network(x)
    
    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))