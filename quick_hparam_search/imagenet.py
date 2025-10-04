import argparse 
import copy
import datetime
import json
import os
import sys
from collections import defaultdict
from tempfile import mkdtemp
 
import numpy as np
import torch 
from torchvision import models 
import random 

import datasets
from utils import Logger

from methods.tact_utils import get_augmentation, get_PCs, remove_PCs

PRETRAINED_WEIGHT_DICT = {
    'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1,
    'vit_b_32': models.ViT_B_32_Weights.IMAGENET1K_V1,
}
  
def TACT(dataset, model, featurize, classifier, classifier_weight, test_loader, base_augmentation, non_causal_augmentation, num_aug, num_pcs_to_remove, device, record_path):
    model.eval()
    ys = []
    
    max_aug = max(num_aug) -1 
    yhats = {'base':[]}
    record_result = {}
    prototypes = {} # store the averaged projected prototype
    num_data = {} # for moving average of prototype 

    for aug in num_aug:
        yhats[aug] = {}
        prototypes[aug] = {}
        num_data[aug] = 0
        record_result[aug] = {}
        for remove in num_pcs_to_remove:
            yhats[aug][remove] = []
            prototypes[aug][remove] = None 
    
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            # get the inputs
            x, y = x.to(device), y.to(device)
            base_x = [base_augmentation(current_x) for current_x in x]
            base_x = torch.stack(base_x).to(device)
            features = featurize(model, base_x)
            # get all augmentations,  include the original features inside as well 
            features_under_augmentation = [features.detach()]
            for _ in range(max_aug):
                augmented_x = non_causal_augmentation(x)
                feature_under_aug = featurize(model, augmented_x) 
                features_under_augmentation.append(feature_under_aug.detach())
            features_under_augmentation = torch.stack(features_under_augmentation)
            
            for aug in num_aug:
                current_features = features_under_augmentation[:aug]
                current_features = current_features.transpose(0, 1)
                _, V, _ = get_PCs(current_features)
                
                for remove in num_pcs_to_remove:
                    # project prototype
                    model_weight = copy.deepcopy(classifier_weight)
                    model_weight = torch.stack([model_weight for _ in range(features.size(0))])
                    projected_prototype = remove_PCs(model_weight, V, 0, remove)
                    # project feature
                    update_f = remove_PCs(features, V, 0, remove)
                    features_to_use = update_f

                    averaged_prototype = torch.mean(projected_prototype, dim=0).cpu()
                    if prototypes[aug][remove] is None:
                        prototypes[aug][remove] = averaged_prototype
                    else:
                        # update prototype using the moving average - only one prototype is kept 
                        prototypes[aug][remove] = (averaged_prototype * x.size(0) + prototypes[aug][remove] * num_data[aug]) / (x.size(0) + num_data[aug])
                    y_hat = torch.mm(features_to_use, prototypes[aug][remove].T.to(device))
                    
                    projection_fn = getattr(dataset, 'project_logits', None)
                    if projection_fn is not None:
                        y_hat = projection_fn(y_hat, device)
                    if isinstance(y_hat, list):
                        y_hat = y_hat[0]

                    yhats[aug][remove].append(y_hat)
                num_data[aug] += x.size(0)
                
            # get the base performance 
            y_hat_base = classifier(model, features)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                y_hat_base = projection_fn(y_hat_base, device)
            if isinstance(y_hat_base, list):
                y_hat_base = y_hat_base[0]
            yhats['base'].append(y_hat_base)
            
            ys.append(y)

        ys = torch.cat(ys).cpu()

        base_preds = torch.cat(yhats['base']).argmax(dim=1, keepdim=True).cpu()
        base_performance = base_preds.eq(ys.view_as(base_preds)).sum().item() / ys.size(0)
        record_result['base'] = base_performance
        print(f"base performance: {base_performance}")

        for aug in num_aug:
            for remove in num_pcs_to_remove:
                ypreds = torch.cat(yhats[aug][remove]).argmax(dim=1, keepdim=True).cpu()
                test_val = ypreds.eq(ys.view_as(ypreds)).sum().item() / ys.size(0)
                
                print(f"num_aug {aug} num of PCs to remove {remove}: {test_val}") 
                record_result[aug][remove] = test_val

        with open(record_path, 'a+') as f:
            f.write(json.dumps(record_result) + '\n')


if __name__ == '__main__':
    runId = datetime.datetime.now().isoformat().replace(':', '_')

    parser = argparse.ArgumentParser(description='Adapt a model during test-time on imagenet datasets.')
    # General
    parser.add_argument('--dataset', type=str, default='imagenet', 
                        choices=['imagenet_r', 'imagenetv2'])
    parser.add_argument('--model', type=str, default='vit_b_32', 
                        choices=['vit_b_32', 'vit_b_16'])
    parser.add_argument('--data_dir', type=str,
                         default='/path/to/data_dir/', help='path to data dir')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA use')
    parser.add_argument('--seed', type=int, default=-1, help='random seed, set as -1 for random.')
    parser.add_argument("--eval_batch_size", default=256, type=int)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args_dict = args.__dict__
    args = argparse.Namespace(**args_dict)

    # Choosing and saving a random seed for reproducibility
    if args.seed == -1:
        args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    # experiment directory setup
    directory_name = f'tact_quick_search/{args.dataset}/{args.model}'
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    run_path = mkdtemp(prefix=runId, suffix=f"_seed{args.seed}", dir=directory_name)
    
    record_path = '{}/run.jsonl'.format(run_path)
    
    # logging setup
    sys.stdout = Logger('{}/run.log'.format(run_path))
    print('RunID:' + run_path)
    with open('{}/args.json'.format(run_path), 'w') as fp:
        json.dump(args.__dict__, fp)
    torch.save(args, '{}/args.rar'.format(run_path))

    agg = defaultdict(list)
    
    # include args also in hparams
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print('experiment seed:' + str(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load dataset
    dataset_class = getattr(datasets, args.dataset)
    n_class = getattr(datasets, 'imagenet_n_class')


    # load model 
    network_class = getattr(models, args.model)
    # by default, model pretrained on imagenet is loaded
    model = network_class(weights=PRETRAINED_WEIGHT_DICT[args.model]).to(device)
    
    def classifier(model, x):
            return model.heads(x)
        
    def featurize(model, x):  
        x = model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x
    
    classifier_weight = model.heads[0].weight.detach()
        
        
    base_augmentation, non_causal_augmentation = get_augmentation(args_dict, device)
    num_aug = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    num_pcs_to_remove = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    test_loader = dataset_class.getTestLoader(args, device)
    TACT(dataset_class, model, featurize, classifier, classifier_weight, test_loader, base_augmentation, non_causal_augmentation, num_aug, num_pcs_to_remove, device, record_path)


    