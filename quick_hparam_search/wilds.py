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
import torch.nn.functional as F
import random 

import datasets, networks 
from config import dataset_defaults
from utils import Logger, return_predict_fn, return_criterion
from test_time_adapt_wilds import load_model
from methods.tact_utils import get_augmentation, get_PCs, remove_PCs


def TACT(args, model, test_loader, base_augmentation, non_causal_augmentation, num_aug, num_pcs_to_remove, device, record_path):
    model.eval()
    ys, metas = [], []
    
    max_aug = max(num_aug) -1 
    yhats = {'base':[]}
    record_result = {}
    prototypes = {} # store the averaged projected prototype
    num_data = {} # for moving average of prototype 
    for aug in num_aug:
        yhats[aug] = {}
        record_result[aug] = {}
        prototypes[aug] = {}
        num_data[aug] = 0
        for remove in num_pcs_to_remove:
            yhats[aug][remove] = []
            prototypes[aug][remove] = None 

    
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            base_x = [base_augmentation(current_x) for current_x in x]
            base_x = torch.cat(base_x).view(x.size()).to(device)
            features = model.featurizer(base_x)
            if args.dataset == 'civil':
                features = model.pre_classifier(features)
                features = F.relu(features)

            # get all augmentations,  include the original features inside as well 
            features_under_augmentation = []
            features_under_augmentation.append(features.detach())
            for _ in range(max_aug): 
                augmented_x = [non_causal_augmentation(current_x) for current_x in x]
                augmented_x = torch.stack(augmented_x).to(device)
                feature_under_aug = model.featurizer(augmented_x) 
                if args.dataset == 'civil':
                    feature_under_aug = model.pre_classifier(feature_under_aug)
                    feature_under_aug = F.relu(feature_under_aug)
                features_under_augmentation.append(feature_under_aug.detach())
            features_under_augmentation = torch.stack(features_under_augmentation)
            
            for aug in num_aug:
                current_features = features_under_augmentation[:aug]
                current_features = current_features.transpose(0, 1)
                _, V, _ = get_PCs(current_features)
                for remove in num_pcs_to_remove:
                    # project prototype 
                    model_weight = copy.deepcopy(model.classifier.weight.detach())
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
                    
                    yhats[aug][remove].append(y_hat)
                num_data[aug] += x.size(0)
                
            # get the base performance 
            y_hat_base = model.classifier(features)
            yhats['base'].append(y_hat_base)

            ys.append(y)
            metas.append(batch[2]) 

        ys, metas = torch.cat(ys).cpu(), torch.cat(metas)
        
        base_performance = test_loader.dataset.eval(predict_fn(torch.cat(yhats['base'])).cpu(), ys, metas)
        record_result['base'] = base_performance[0][args.selection_metric]
        print(f"=============== base performance ==============\n{base_performance[-1]}")
        

        for aug in num_aug:
            for remove in num_pcs_to_remove:
                ypreds  = predict_fn(torch.cat(yhats[aug][remove]))
                test_val = test_loader.dataset.eval(ypreds.cpu(), ys, metas)
                print(f"=============== num_aug: {aug} num_pcs_to_remove: {remove}===============\n{test_val[-1]}")        

                record_result[aug][remove] = test_val[0][args.selection_metric]

        with open(record_path, 'a+') as f:
            f.write(json.dumps(record_result) + '\n')


if __name__ == '__main__':
    runId = datetime.datetime.now().isoformat().replace(':', '_')

    parser = argparse.ArgumentParser(description='Quick hyperparameter search of TACT for WILDS datasets.')
    # General
    parser.add_argument('--dataset', type=str, default='camelyon',
                    help="Name of dataset, choose from birdcalls, camelyon")
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--data_dir', type=str, default='/path/to/data_dir/',
                        help='path to data dir')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA use')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed, set as -1 for random.')
    parser.add_argument("--eval_batch_size", default=256, type=int) # batch size matters here 
    parser.add_argument('--use_published_model', action="store_true")
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args_dict = args.__dict__
    args_dict.update(dataset_defaults[args.dataset])
    args = argparse.Namespace(**args_dict)

    # Choosing and saving a random seed for reproducibility
    if args.seed == -1:
        args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    
    model_kwargs = {}
    if args.model in ['vitb32', 'vitb16']:
        if args.dataset == 'camelyon':
            image_size = 96 
        elif args.dataset == 'birdcalls':
            image_size = 224
        model_kwargs['image_size'] = image_size
    
    # experiment directory setup
    base_model_name = args.model_path.split('.')[0]
    directory_name = f'tact_quick_search/{args.dataset}/{base_model_name}'
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
    args.num_workers = 4

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print('experiment seed:' + str(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load dataset
    datasetC = getattr(datasets, args.dataset)
    test_loader = datasetC.getTestLoader(args, device)
    n_class = getattr(datasets, f"{args.dataset}_n_class")

    # load model 
    networkC = getattr(networks, args.model)
    print('Loading base model', args.model_path)
    model = load_model(args, networkC, n_class, model_kwargs, device)
    predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)

    base_augmentation, non_causal_augmentation = get_augmentation(args_dict, device)
    num_aug = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    num_pcs_to_remove = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    TACT(args, model, test_loader, base_augmentation, non_causal_augmentation, num_aug, num_pcs_to_remove, device, record_path)

    