import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .tact_utils import get_augmentation, get_PCs, remove_PCs
from .utils import softmax_entropy

class CT(nn.Module): # only trim representation 
    def __init__(self, model, num_classes, hparams, device):
        super().__init__()
        self.num_aug = hparams['num_aug']
        self.start_pc_to_remove = hparams['start_pc']
        self.num_pcs_to_remove = hparams['num_pcs']

        self.model = model
        self.featurizer = model.network[:-1]
        self.classifier = model.network[-1]

        self.base_augmentation, self.non_causal_augmentation = get_augmentation(hparams['args'], device)

        self.hparams = hparams
        self.num_classes = num_classes
        self.device = device
        self.dataset = hparams['args']['dataset']  

    def forward(self, x, adapt=False):
        features = self.get_features(x)
        if adapt:
            update_f = self.causal_trimming(x, features)
            return self.output_probability(update_f)
        else:
            return self.output_probability(features)
    
    def get_features(self, x): # perform base augmentation (i.e. normalization) to get features
        base_x = [self.base_augmentation(current_x) for current_x in x]
        base_x = torch.stack(base_x).to(self.device)
        features = self.featurizer(base_x)
        return features
    
    def causal_trimming(self, x, features):
        # store features
        all_features = [features.detach()]
        
        # get features under non causal augmentation 
        for _ in range(self.num_aug):
            # when augmentation can only apply to every x one by one 
            if self.dataset in ['camelyon', 'birdcalls', 'civil']:
                augmented_x = [self.non_causal_augmentation(current_x) for current_x in x]
                augmented_x = torch.stack(augmented_x).to(self.device)
            else:
                augmented_x = self.non_causal_augmentation(x)
            feature_under_augmentation = self.featurizer(augmented_x) 
            all_features.append(feature_under_augmentation.detach())  

        all_features = torch.stack(all_features)
        all_features = all_features.transpose(0, 1)
        
        # eigendecomposition to find PCs
        _, V, _ = get_PCs(all_features)
        
        # remove PCs from features
        update_f = remove_PCs(features, V, self.start_pc_to_remove, self.num_pcs_to_remove)
        return update_f.to(self.device)
    
    def output_probability(self, features):
        return self.classifier(features)


class TACT(CT):
    def __init__(self, model, num_classes, hparams, device):
        super().__init__(model, num_classes, hparams, device)
        self.prototypes = self.classifier.weight.data
        self.num_samples_seen = 0
        self.updated_prototypes = None 

    # update feature as well as prototypes
    def causal_trimming(self, x, features): 
        # store features
        all_features = [features.detach()]
        
        # get features under non causal augmentation 
        for _ in range(self.num_aug):
            # when augmentation can only apply to every x one by one 
            if self.dataset in ['camelyon', 'birdcalls', 'civil']:
                augmented_x = [self.non_causal_augmentation(current_x) for current_x in x]
                augmented_x = torch.stack(augmented_x).to(self.device)
            else:
                augmented_x = self.non_causal_augmentation(x)
            feature_under_augmentation = self.featurizer(augmented_x) 
            all_features.append(feature_under_augmentation.detach())  

        all_features = torch.stack(all_features)
        all_features = all_features.transpose(0, 1)
        
        # eigendecomposition to find PCs
        _, V, _ = get_PCs(all_features)
        
        # remove PCs from features
        update_f = remove_PCs(features, V, self.start_pc_to_remove, self.num_pcs_to_remove)

        # update prototype
        model_prototype = copy.deepcopy(self.prototypes)
        model_prototype = torch.stack([model_prototype for _ in range(features.size(0))])
        
        updated_prototype = remove_PCs(model_prototype, V, self.start_pc_to_remove, self.num_pcs_to_remove)
        
        averaged_prototype = torch.mean(updated_prototype, dim=0)
        if self.updated_prototypes is None:
            self.updated_prototypes = averaged_prototype
        else:
            self.updated_prototypes = (self.updated_prototypes * self.num_samples_seen + averaged_prototype * features.size(0)) / (self.num_samples_seen +  features.size(0))

        self.num_samples_seen += features.size(0)

        return update_f.to(self.device)
    
    # use updated prototype to make prediction 
    def output_probability(self, features):
        return torch.mm(features,  self.updated_prototypes.T)
    

class TACT_adapt(TACT):
    def __init__(self, model, num_classes, hparams, device):
        super().__init__(model, num_classes, hparams, device)
        self.adapt_model = copy.deepcopy(model)

        self.entropy_weighting = hparams['entropy_weighting']

        self.optimizer = torch.optim.Adam(
                self.adapt_model.network[:-1].parameters(),
                lr=hparams['adaptation_lr'],
            )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, adapt=False):
        base_x = [self.base_augmentation(current_x) for current_x in x]
        base_x = torch.stack(base_x).to(self.device)
        
        if adapt:
            features = self.featurizer(base_x)
            update_f = self.causal_trimming(x, features)
            prediction = self.output_probability(update_f)
            return self.forward_and_adapt(base_x, self.adapt_model, self.optimizer, prediction)
        else:
            return self.adapt_model(base_x)
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer, prediction):
        output = model(x)
        hard_loss = self.criterion(output, prediction.argmax(1))
        entropy = softmax_entropy(output).mean(0)
        
        softmax_out = F.softmax(output, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        entropy += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        loss =  hard_loss + self.entropy_weighting *entropy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output
