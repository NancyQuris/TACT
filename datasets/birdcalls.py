import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from wilds.common.data_loaders import get_eval_loader

from .wilds_datasets import BirdCallsDataset, GeneralWilds_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 32
PRETRAINED = True

class birdcalls:
    def __init__(self):
        self.num_classes = NUM_CLASSES
        self.pretrained = PRETRAINED

    @staticmethod
    def getTrainData(args):
        dataset = BirdCallsDataset(root_dir=args.data_dir, split_scheme=args.split_scheme, download=True)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = dataset.get_subset('train', transform=transform)
        train_sets = GeneralWilds_Batched_Dataset(args, train_data, args.batch_size, domain_idx=0)
        return train_sets 
    
    @staticmethod
    def getTrainLoader(args, device):
        train_sets = birdcalls.getTrainData(args)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loaders
    
    @staticmethod
    def getTestLoader(args, device, val_domain=False):
        dataset = BirdCallsDataset(root_dir=args.data_dir, split_scheme=args.split_scheme, keep_ood_train=True, download=True)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if val_domain:
            test_data = dataset.get_subset('ood_val', transform=transform)
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
        dataset = BirdCallsDataset(root_dir=args.data_dir, split_scheme=args.split_scheme, keep_ood_train=True, download=True)
        '''
        if keep_ood_train is False dataset length is:
            train 2089, val 0, test 724, id_val 407, id_test 1677, ood_train 0, ood_val 0
        if keep_ood_train is True dataset length is:
            train 2089, val 0, test 290, id_val 407, id_test 1677, ood_train 365, ood_val 69
        '''
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # get all train data
        train_data = dataset.get_subset('train', transform=transform)
        
        # # separate into subsets by distribution
        train_sets = train_data

        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, dataset in datasets.items():
            if split in split_tv:
                # split the dataset to 20% for train and 80% for validation
                train_size = int(0.2 * len(dataset))
                val_size = len(dataset) - train_size
                train_subset, val_subset = random_split(dataset, [train_size, val_size])
                
                tv_loaders[f'{split}_train'] = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **kwargs)
                tv_loaders[f'{split}_val'] = DataLoader(val_subset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)
            else:
                if shuffle_test:
                    if len(dataset) == 0:
                        continue
                    else: 
                        tv_loaders[split] = DataLoader(dataset, 
                                            batch_size=args.eval_batch_size, 
                                            shuffle=True, 
                                            sampler=None,
                                            collate_fn=dataset.collate,
                                            **kwargs)
                else:
                    tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.eval_batch_size)
        
        return train_loaders, tv_loaders
