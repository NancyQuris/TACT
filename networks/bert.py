from copy import deepcopy
import torch.nn as nn

from transformers import BertModel

class BertFeaturizer(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1] # get pooled output
        return outputs
    
class Network(nn.Module):
    def __init__(self, num_classes, pretrained=True, group_num=None, weights=None):
        super(Network, self).__init__()
        self.featurizer = BertFeaturizer.from_pretrained("bert-base-uncased")

        if group_num is not None:  
            for name, module in self.featurizer.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn = getattr(self.featurizer, name)
                    gn = nn.GroupNorm(group_num, bn.num_features)
                    nn.init.constant_(gn.weight, 1)
                    nn.init.constant_(gn.bias, 0)
                    setattr(self.featurizer, name, gn)

        self.n_output = self.featurizer.config.hidden_size
        self.pre_classifier = nn.Linear(self.n_output, self.n_output)
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