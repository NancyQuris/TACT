import argparse
import datetime
import itertools
import json
import os
import sys
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import random 

 
import datasets, methods, networks
from utils import Logger
from adaptation_config import get_adapt_hparams_list
from test_time_adapt_wilds import get_algorithm

''' test model performance (no adaptation) '''
def test(model, dataset, test_loader):
    model.eval()
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, (inputs, y) in enumerate(test_loader):
            inputs = inputs.to(device)
            y = y.to(device)
            
            logits = model(inputs)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            if i % 100 == 0:
                percent_complete = 100.0 * i / len(test_loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(test_loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}"
                )
        top1 = correct / n
        print(f"==============================\nTop1 acc: {top1}")
        return top1
    
''' adapt model and return performance '''
def adapt(model, dataset, test_loader):
    model.eval()
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, (inputs, y) in enumerate(test_loader):
            inputs = inputs.to(device)
            y = y.to(device)
            if args.algorithm in ['LAME', 'FOA', 'SHOT', 'SAR', 'DeYO', 'TSD', 'PASLE']:
                logits = model(inputs)
            else:
                logits = model(inputs, adapt=True)
            
            
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            if i % 100 == 0:
                percent_complete = 100.0 * i / len(test_loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(test_loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}"
                )
        top1 = correct / n
        print(f"==============================\n{top1}")

        return top1

if __name__ == '__main__':
    runId = datetime.datetime.now().isoformat().replace(':', '_')

    parser = argparse.ArgumentParser(description='Adapt a model during test-time on ImageNet.')
    # General
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet_r', 'imagenetv2'])
    parser.add_argument('--algorithm', type=str, default='tent',
                        help='training scheme, choose between fish or erm.')
    parser.add_argument('--experiment', type=str, default='.',
                        help='experiment name, set as . for automatic naming.')
    parser.add_argument('--data_dir', type=str, default='/path/to/data_dir/',
                        help='path to data dir')
    parser.add_argument('--model', type=str, default='vit_b_32', choices=['vit_b_32', 'vit_b_16'])
    # Computation
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA use')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed, set as -1 for random.')
    parser.add_argument("--eval_batch_size", default=256, type=int)
    
    # for training - no effect for TTA 
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    
   
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args_dict = args.__dict__
    args_dict['optimiser_args'] = {
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    args = argparse.Namespace(**args_dict)

    # Choosing and saving a random seed for reproducibility
    if args.seed == -1:
        args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    # experiment directory setup
    args.experiment = f"{args.dataset}/bs{args.eval_batch_size}/{args.algorithm}/{args.model}" \
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
    
    best_adapt_hparams = None
    best_test = 0 
 
    for adapt_hparams in adapt_hparams_list:
        record_result = {'hparam': adapt_hparams}
        print(adapt_hparams)

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
        dataset_class = getattr(datasets, args.dataset)
        
        # by default, only the validation set is returned 
        if args.algorithm == 'base':
            tv_loaders = dataset_class.getDataLoaders(args, device, shuffle_test=True)
            # train_loader = dataset_class.getTrainLoader(args, device)
            train_loader = tv_loaders['id_val']
            test_loader = tv_loaders['val']
            
            model = networks.imagenet_network(args.model).to(device)
            algorithm = model
            test_adapt_result = test(algorithm, dataset_class, test_loader)

            print(f'no adaptation performance {test_adapt_result}')

        else:
            tv_loaders = dataset_class.getDataLoaders(args, device, shuffle_test=True)
            train_loader = dataset_class.getTrainLoader(args, device)
            
            test_loader = tv_loaders['val']
            if args.algorithm in ['TACT', 'TACT_adapt']:
                test_loader = dataset_class.getTestLoader(args, device)
            
            n_class = getattr(datasets, 'imagenet_n_class')
            model = networks.imagenet_network(args.model).to(device)    

            ''' adapt on test domain, create new model to test performance'''
            algorithm = get_algorithm(args, model, adapt_hparams, n_class, train_loader, tv_loaders, device)
            if args.algorithm == 'FOA':
                algorithm.obtain_origin_stat(train_loader)
            test_adapt_result = adapt(algorithm, dataset_class, test_loader)
            
            record_result['test'] = test_adapt_result

            if test_adapt_result > best_test:
                best_test = test_adapt_result
                best_adapt_hparams = adapt_hparams


            with open(record_path, 'a+') as f:
                f.write(json.dumps(record_result) + '\n')
        
    print(f'best adapt hparams are {best_adapt_hparams}, performance on test domain is {best_test}')