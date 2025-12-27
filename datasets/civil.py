import torch
import numpy as np
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_eval_loader
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from transformers import DistilBertTokenizerFast
from .wilds_datasets import CivilComments_Batched_Dataset

MAX_TOKEN_LENGTH = 300
NUM_CLASSES = 2
PRETRAINED = True

def initialize_bert_transform(args):
    model = args.model
    if model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt')
        if model == 'bert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        else:
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform


class civil:
    def __init__(self):
        self.num_classes = NUM_CLASSES

    @staticmethod
    def getTrainData(args):
        dataset = CivilCommentsDataset(root_dir=args.data_dir, download=True)
        transform = initialize_bert_transform(args)
        train_data = dataset.get_subset('train', transform=transform)
        train_sets = CivilComments_Batched_Dataset(args, train_data, batch_size=args.batch_size)
        return train_sets
    
    @staticmethod
    def getTrainLoader(args, device):
        train_sets = civil.getTrainData(args)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        
        if args.reweight_groups:
            print(f"upweighting wrong groups by factor {args.upweight_factor}")
            assert args.group_by_wrong
            assert train_sets.num_envs == 2
            group_weights = np.array([args.upweight_factor, 1])
            print(f"Wrong: weight = {group_weights[0]}, numbers = {len(np.where(train_sets.domains == 0)[0])}")
            print(f"Correct: weight = {group_weights[1]}, numbers = {len(np.where(train_sets.domains == 1)[0])}")
            weights = group_weights[train_sets.domains]

            assert len(weights) == len(train_sets)
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)
        else:
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loaders
    
    @staticmethod
    def getTestLoader(args, device, val_domain=False):
        dataset = CivilCommentsDataset(root_dir=args.data_dir, download=True)
        transform = initialize_bert_transform(args)
        if val_domain:
            test_data = dataset.get_subset('val', transform=transform)
        else:
            test_data = dataset.get_subset('test', transform=transform)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        test_loader = DataLoader(test_data,
                                batch_size=args.eval_batch_size, 
                                shuffle=True, 
                                sampler=None,
                                collate_fn=dataset.collate,
                                **kwargs)
        return test_loader
    
    @staticmethod
    def getDataLoaders(args, device, shuffle_test=False, split_tv=[]):
        dataset = CivilCommentsDataset(root_dir=args.data_dir, download=True)
        transform = initialize_bert_transform(args)
        train_data = dataset.get_subset('train', transform=transform)
        
        train_sets = train_data 

        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)
        
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        
        if args.reweight_groups:
            print(f"upweighting wrong groups by factor {args.upweight_factor}")
            assert args.group_by_wrong
            assert train_sets.num_envs == 2
            group_weights = np.array([args.upweight_factor, 1])
            print(f"Wrong: weight = {group_weights[0]}, numbers = {len(np.where(train_sets.domains == 0)[0])}")
            print(f"Correct: weight = {group_weights[1]}, numbers = {len(np.where(train_sets.domains == 1)[0])}")
            weights = group_weights[train_sets.domains]

            assert len(weights) == len(train_sets)
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)
        else:
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)

        tv_loaders = {}
        for split, dataset in datasets.items():
            if split in split_tv:
                train_size = int(0.2*len(dataset))
                val_size = len(dataset) - train_size
                train_subset, val_subset = random_split(dataset, [train_size, val_size])
            
                tv_loaders[f'{split}_train'] = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **kwargs)
                tv_loaders[f'{split}_val'] = DataLoader(val_subset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)
            else:
                if shuffle_test:
                    tv_loaders[split] = DataLoader(dataset,
                                        batch_size=args.eval_batch_size, 
                                        shuffle=True, 
                                        sampler=None,
                                        collate_fn=dataset.collate,
                                        **kwargs)
                else:
                    tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.eval_batch_size)
        return train_loaders, tv_loaders


