import argparse
import datetime
import itertools
import json
import os
import sys
import csv
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import random 

import datasets, networks 
import methods
from config import dataset_defaults
from utils import Logger, return_predict_fn, return_criterion, map_dict
from adaptation_config import get_adapt_hparams_list

''' test model performance (no adaptation) '''
def test(model, test_loader, predict_fn, run_path, agg, args, loader_type='test', verbose=True, save_ypred=False, subset=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                        f"{args.seed}_epoch:best_pred.csv"
            with open(f"{run_path}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        if subset:
            test_val = test_loader.dataset.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        else:
            test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")

        return test_val

''' adapt model and return performance '''
def adapt(model, test_loader, predict_fn, run_path, agg, args, loader_type='test', verbose=True, save_ypred=False, subset=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            if args.algorithm in  ['LAME', 'FOA', 'SHOT', 'SAR', 'DeYO', 'TSD', 'PASLE']:
                y_hat = model(x)
            else:
                y_hat = model(x, adapt=True)
            ys.append(y.cpu())
            yhats.append(y_hat.cpu())
            metas.append(batch[2])
        
        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                        f"{args.seed}_epoch:best_pred.csv"
            with open(f"{run_path}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        if subset:
            test_val = test_loader.dataset.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        else:
            test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")
        

     
        return test_val

''' get algorithm for adaptation '''
def get_algorithm(args, model, adapt_hparams, n_class, train_loader, tv_loaders, device):
    # get algorithm class
    algorithm_class = methods.get_algorithm_class(args.algorithm)   
    
    if args.algorithm in ['TACT', 'TACT_adapt']:
        algorithm = algorithm_class(model, n_class, adapt_hparams, device)
    elif args.algorithm == 'LAME':
        dataset_class = getattr(datasets, args.dataset)
        projection_fn = getattr(dataset_class, 'project_logits', None)
        algorithm = algorithm_class(model, n_class, adapt_hparams, projection_fn=projection_fn)
    else:
        algorithm = algorithm_class(model, n_class, adapt_hparams)
    
    return algorithm
             

def load_model(args, networkC, n_class, model_kwargs, device):
    if args.use_published_model:
        pretrained_model = map_dict(torch.load(args.model_path)['algorithm'], args.dataset)
        model = networkC(n_class, pretrained=False, group_num=args.group_num, weights=pretrained_model).to(device)
    else:
        model = networkC(n_class, pretrained=False, group_num=args.group_num, weights=None, **model_kwargs).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    return model

if __name__ == '__main__':
    runId = datetime.datetime.now().isoformat().replace(':', '_')

    parser = argparse.ArgumentParser(description='Adapt a model during test-time on wilds datasets.')
    # General
    parser.add_argument('--dataset', type=str, default='camelyon',
                    help="Name of dataset, choose from birdcalls, camelyon")
    parser.add_argument('--algorithm', type=str, default='erm',
                        help='training scheme, choose between fish or erm.')
    parser.add_argument('--experiment', type=str, default='.',
                        help='experiment name, set as . for automatic naming.')
    parser.add_argument('--data_dir', type=str, default='/path/to/data_dir/',
                        help='path to data dir')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA use')
    parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument('--model_path', type=str, default=None, help='path to pretrained model')
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
    if args.model == 'vitb32':
        if args.dataset == 'camelyon':
            image_size = 96 
        elif args.dataset == 'birdcalls':
            image_size = 224
        model_kwargs['image_size'] = image_size

    # experiment directory setup
    if args.use_published_model:
        base_model_name = args.model_path.split('/')[-1].split('.')[0]
    else:
        base_model_name = args.model_path.split('/')[-2].split('.')[0]
    args.experiment = f"{args.dataset}/bs{args.eval_batch_size}/{args.algorithm}/{base_model_name}" \
        if args.experiment == '.' else args.experiment
    
    directory_name = 'experiments_tta/{}'.format(args.experiment)
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
    print(
        "=" * 30 + f"Adapting: {args.algorithm}" + "=" * 30)
    
    adapt_hparams_dict = get_adapt_hparams_list(args)
    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]
    
    best_test_val = 0
    best_test_by_test_val = 0 
    best_adapt_hparams_by_test_val = None

    for adapt_hparams in adapt_hparams_list:
        record_result = {'hparam': adapt_hparams}
        print(adapt_hparams)

        args.num_workers = 4
        adapt_hparams['args'] = args.__dict__

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
        train_loader, tv_loaders = datasetC.getDataLoaders(args, device=device, shuffle_test=True)
        
        if args.algorithm in ['TACT', 'TACT_adapt']: 
            test_loader = datasetC.getTestLoader(args, device=device)
            val_loader = datasetC.getTestLoader(args, device=device, val_domain=True)
            val_ood_type = 'val' if args.dataset != 'birdcalls' else 'ood_val'
        else:
            test_loader = tv_loaders['test']
            if args.dataset != 'birdcalls':
                val_loader = tv_loaders['val']
                val_ood_type = 'val'
            else:
                val_loader = tv_loaders['ood_val']
                val_ood_type = 'ood_val'
        
        n_class = getattr(datasets, f"{args.dataset}_n_class")
        predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)
        networkC = getattr(networks, args.model)
        
        # load model 
        test_model = load_model(args, networkC, n_class, model_kwargs, device)
        ''' adapt on test domain, create new model to test performance  '''
        if args.algorithm == 'base':
            test_algorithm = test_model
            test_adapt_results = test(test_algorithm, test_loader, predict_fn, run_path, agg, args, loader_type='test')
        else:
            test_algorithm = get_algorithm(args, test_model, adapt_hparams, n_class, train_loader, tv_loaders, device)
            if args.algorithm == 'FOA':
                test_algorithm.obtain_origin_stat(train_loader)
            test_adapt_results = adapt(test_algorithm, test_loader, predict_fn, run_path, agg, args, loader_type='test')  
        record_result['test'] = test_adapt_results[0]

        
        test_val_eval_result = test_adapt_results[0][args.selection_metric]
        if test_val_eval_result > best_test_val:
            best_adapt_hparams_by_test_val = adapt_hparams
            best_test_by_test_val = test_adapt_results[0][args.selection_metric]
            best_test_val = test_val_eval_result


        with open(record_path, 'a+') as f:
            f.write(json.dumps(record_result) + '\n')
    
    print(f'select by oracle: best adapt hparams are {best_adapt_hparams_by_test_val}, performance on test domain is {best_test_by_test_val}')