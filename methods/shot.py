'''
    from https://github.com/tim-learn/SHOT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import softmax_entropy, copy_model_and_optimizer, load_model_and_optimizer

class SHOT(nn.Module):
    def __init__(self, model, num_classes, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes

        self.beta = hparams['beta']
        self.theta = hparams['theta'] 

        self.model = model
        self.featurizer = model.network[:-1]
        self.optimizer = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=hparams["adaptation_lr"],)
        self.steps = hparams['adaptation_step']
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = hparams['episodic']

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()
        return outputs
    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

        loss = ent_loss + self.theta * clf_loss
        return loss

class SHOTIM(SHOT):
    def __init__(self, model, num_classes, hparams):
        super().__init__(model, num_classes, hparams)
        self.hparams = hparams
        self.num_classes = num_classes

        self.model = model
        self.featurizer = model.network[:-1]
        self.optimizer = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=hparams["adaptation_lr"],)
        self.steps = hparams['adaptation_step']
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = hparams['episodic']

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model(x)
        # adapt
        ent_loss = softmax_entropy(outputs).mean(0)
        softmax_out = F.softmax(outputs, dim=1)
        msoftmax = softmax_out.mean(dim=0)
        div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = ent_loss + div_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs