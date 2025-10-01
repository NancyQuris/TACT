'''
    from https://github.com/Jhyun17/DeYO
'''

import torch 
import torch.nn as nn
import torchvision
import copy
import math 
from einops import rearrange


class DeYO(nn.Module):
    def __init__(self, model, num_classes, hparams):
        super().__init__()
        self.model = configure_model(model)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.SGD(
            params,
            lr=hparams['adaptation_lr'],
            momentum=hparams['adaptation_momentum']
        )

        self.num_classes = num_classes
        self.episodic = hparams['episodic']
        self.steps = hparams['adaption_step']

        self.deyo_margin = hparams['deyo_margin'] * math.log(num_classes)
        self.margin_e0 = hparams['margin_e0'] * math.log(num_classes)

        self.hparams = hparams

    def configure_model_optimizer(self, model, hparams):
        adapted_algorithm = copy.deepcopy(model)
        adapted_algorithm = configure_model(adapted_algorithm)
        params, param_names = collect_params(adapted_algorithm)
        
        opt = getattr(torch.optim, hparams['optimizer'])
        optimizer = opt(
            params, 
            lr=hparams['adaptation_lr'],
            weight_decay=hparams['adaptation_wd'],
            momentum=hparams['adaptation_momentum']
        )
        return adapted_algorithm, optimizer
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def forward(self, x, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_deyo(x, self.model, self.hparams,
                                                                              self.optimizer, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group)
                else:
                    outputs = forward_and_adapt_deyo(x, self.model, self.hparams,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_deyo(x, self.model, 
                                                                                                    self.hparams, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group)
                else:
                    outputs = forward_and_adapt_deyo(x, self.model, 
                                                    self.hparams, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group, self)
        return outputs
        
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt_deyo(x, model, hparams, optimizer, deyo_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    if hparams['filter_ent']:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if hparams['aug_type']=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, hparams['occlusion_size'], hparams['occlusion_size'])
        x_prime[:, :, hparams['row_start']:hparams['row_start']+hparams['occlusion_size'], \
                hparams['column_start']:hparams['column_start']+hparams['occlusion_size']] = occlusion_window
    elif hparams['aug_type']=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//hparams['patch_len'])*hparams['patch_len'],(x.shape[-1]//hparams['patch_len'])*hparams['patch_len']))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=hparams['patch_len'], ps2=hparams['patch_len'])
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=hparams['patch_len'], ps2=hparams['patch_len'])
        x_prime = resize_o(x_prime)
    elif hparams['aug_type']=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if hparams['filter_plpd']:
        filter_ids_2 = torch.where(plpd > hparams['plpd_threshold'])
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0
        
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if hparams['reweight_ent'] or hparams['reweight_plpd']:
        coeff = (hparams['reweight_ent'] * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 hparams['reweight_plpd'] * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'encoder_layer_9' in nm:
            continue
        if 'encoder_layer_10' in nm:
            continue
        if 'encoder_layer_11' in nm:
            continue
        if 'encoder.ln.' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']: 
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with DeYO."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model