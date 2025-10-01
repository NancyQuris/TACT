import argparse
import datetime
import json
import os
import sys
import csv
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim

import datasets, networks 
from config import dataset_defaults
from utils import unpack_data,  save_best_model, Logger, return_predict_fn, return_criterion

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Pretrain a model for test-time adaptation.')
# General
parser.add_argument('--dataset', type=str, default='camelyon',
                    help="Name of dataset, choose from birdcalls, camelyon")
parser.add_argument('--algorithm', type=str, default='erm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data_dir', type=str, default='/path/to/data_dir/',
                    help='path to data dir')


# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')
parser.add_argument("--n_groups_per_batch", default=4, type=int)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--save_dir", default='result', type=str)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
print(args.seed)
torch.backends.cudnn.deterministic = True

model_kwargs = {}
if args.dataset == 'camelyon':
    image_size = 96 
elif args.dataset == 'birdcalls':
    image_size = 224
model_kwargs['image_size'] = image_size

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
directory_name = 'experiments/{}'.format(args.experiment)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
datasetC = getattr(datasets, args.dataset)
if args.dataset == 'camelyon': args.n_groups_per_batch = 3
_, tv_loaders = datasetC.getDataLoaders(args, device=device)
train_loader = datasetC.getTrainLoader(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
n_class = getattr(datasets, f"{args.dataset}_n_class")
pretrained = getattr(datasets, f"{args.dataset}_pretrained")

networkC = getattr(networks, args.model)
model = networkC(n_class, pretrained=pretrained, group_num=args.group_num, weights=None, **model_kwargs).to(device)

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)
optimiserC = opt(model.parameters(), **args.optimiser_args)

predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)


def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()

        optimiserC.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])
        
        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        
        if save_ypred:
            save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                            f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")


if __name__ == '__main__':
    scheduler = None

    print(
        "=" * 30 + f"Training: {args.algorithm}" + "=" * 30)
    train = locals()[f'train_{args.algorithm}']
    agg = defaultdict(list)
    if args.dataset == 'birdcalls':
        agg['id_val_stat'] = [0.]
    else:
        agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    for epoch in range(args.epochs):
        train(train_loader, epoch, agg)
        if args.dataset == 'birdcalls':
            test(tv_loaders['id_val'], agg, loader_type='id_val')
        else:
            test(val_loader, agg, loader_type='val')
        test(test_loader, agg, loader_type='test')
        save_best_model(model, runPath, agg)
        
    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        if len(loader) > 0:
            test(loader, agg, loader_type=split, save_ypred=True)