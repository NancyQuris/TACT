'''
    from https://github.com/DequanWang/tent
'''

import torch
import torch.nn as nn
import copy

from .utils import copy_model_and_optimizer, configure_model, collect_params, load_model_and_optimizer
from .utils import softmax_entropy

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, num_classes, hparams):
        super().__init__()
        self.model, self.optimizer = self.configure_model_optimizer(model, hparams)
        self.steps = hparams['adaptation_step']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = hparams['episodic']
        self.num_classes = num_classes

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def configure_model_optimizer(self, model, hparams):
        adapted_algorithm = copy.deepcopy(model)
        adapted_algorithm = configure_model(adapted_algorithm)
        params, param_names = collect_params(adapted_algorithm)
        
        opt = getattr(torch.optim, hparams['optimizer'])
        optimizer = opt(
            params, 
            lr=hparams['adaptation_lr'],
            momentum=hparams['adaptation_momentum']
        )

        self.params = params 

        return adapted_algorithm, optimizer
    
    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        else:
            outputs = self.model(x)
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        