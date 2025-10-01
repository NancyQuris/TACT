'''
    from https://github.com/palm-ml/PASLE
'''

from copy import deepcopy
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .utils import collect_params, softmax_entropy

class PASLE(nn.Module):
    def __init__(self, model, num_classes, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes

        self.featurizer = model.network[:-1]
        self.classifier = model.network[-1]

        self.model = model

        opt = getattr(torch.optim, hparams['optimizer'])
        
        if hparams['param_to_adapt'] == 'all':
            params = self.model.parameters()
        elif hparams['param_to_adapt'] == 'head':
            params = self.classifier.parameters()
        elif hparams['param_to_adapt'] == 'body':
            params = self.featurizer.parameters()
        elif hparams['param_to_adapt'] == 'affine':
            self.model.train()
            self.model.requires_grad_(False)
            params,_ = collect_params(self.model)
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    m.requires_grad_(True)
        
        self.optimizer = opt(
            params, 
            lr=hparams["adaptation_lr"],
        )
        
        self.thresh = hparams['thresh']
        self.thresh_end = hparams['thresh'] - hparams['thresh_gap']
        self.thresh_des = hparams['thresh_des']
        self.temp = hparams['temp']
        self.buffer_size = hparams['buffer_size']     

        self.samples_buffer = None

        # enhance module
        self.model_copy = deepcopy(self.model).eval()

        self.filter_K = 100
               
        warmup_supports = self.model_copy.network[-1].weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model_copy.network[-1](self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
    
    @torch.enable_grad()
    def forward(self,samples):

        self.origin_sample_num = samples.shape[0]

        if self.samples_buffer != None:
            samples = torch.cat((samples,self.samples_buffer),dim=0)

        logits = self.model(samples)

        probs = F.softmax(logits,1)
        probs_des, _ = torch.sort(probs, descending=True)

        margins = probs_des[:,0] - probs_des[:,1]

        mask_hard = margins > self.thresh
        mask_unselect = (probs_des[:,0] - probs_des[:,-1]) < self.thresh
        mask_partial = ~ (mask_hard | mask_unselect)

        _, idxs = torch.sort(margins[mask_unselect], descending=True)
        if idxs.shape[0] > self.buffer_size:
            self.samples_buffer = samples[mask_unselect][idxs][:self.buffer_size]
        else:
            self.samples_buffer = samples[mask_unselect]
        
        partial_labels = ((probs[mask_partial] + self.thresh) > probs_des[mask_partial][:,0].reshape(-1,1)).long()

        # label filter        
        label_prototype = self.get_label_with_prototype(samples).argmax(1)
        mask_hard_same = logits[mask_hard].argmax(1) == label_prototype[mask_hard]
        rows = torch.arange(mask_partial.long().sum())
        cols = label_prototype[mask_partial]
        mask_partial_same = (partial_labels)[rows,cols] > 0

        loss_hard = nn.CrossEntropyLoss()(logits[mask_hard][mask_hard_same] / self.temp, logits[mask_hard][mask_hard_same].detach().argmax(1))
        loss_partial = cc_loss(logits[mask_partial][mask_partial_same], partial_labels[mask_partial_same].detach(), self.temp)
        lam_hard = mask_hard_same.long().sum() / (mask_hard_same.long().sum() + mask_partial_same.long().sum())
        loss = loss_hard * lam_hard + loss_partial * (1 - lam_hard)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.thresh > self.thresh_end:
            self.thresh -= self.thresh_des
        
        return logits[0:self.origin_sample_num]
    
    @torch.no_grad() 
    def get_label_with_prototype(self,x):
        z = self.model_copy.network[:-1](x)
        p = self.model_copy.network[-1](z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports,z[0:self.origin_sample_num,:]])
        self.labels = torch.cat([self.labels,yhat[0:self.origin_sample_num]])
        self.ent = torch.cat([self.ent,ent[0:self.origin_sample_num]])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

def cc_loss(outputs, partialY, temp):
    sm_outputs = F.softmax(outputs / temp, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss