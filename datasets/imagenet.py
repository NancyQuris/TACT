import torch 
from torchvision.datasets import ImageNet, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from imagenetv2_pytorch import ImageNetV2Dataset

NUM_CLASSES = 1000
PRETRAINED = True 

IMAGENET_R_CLASS_SUBLIST = [
        1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107,
        113, 122,
        125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203,
        207, 208, 219,
        231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
        291, 292, 293,
        296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347,
        353, 355, 361,
        362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447,
        448, 457, 462,
        463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613,
        617, 621, 629,
        637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852,
        866, 875, 883,
        889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
        967, 980, 981,
        983, 988]
    
IMAGENET_R_CLASS_SUBLIST_MASK = [(i in IMAGENET_R_CLASS_SUBLIST) for i in range(1000)]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])

def standard_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def transform_for_augmentation():
    transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform

def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class imagenet:
    def __init__(self):
        self.num_classes = NUM_CLASSES
        self.pretrained = PRETRAINED

    @staticmethod
    def getTrainData(args):
        dataset = ImageNet(root=args.data_dir, split='train', transform=standard_transform())
        return dataset
    
    @staticmethod
    def getTestData(args, use_transforms=False):
        transforms = te_transforms if use_transforms else None
        dataset = ImageNet(root=args.data_dir, split='val', transform=transforms)
        return dataset 

    @staticmethod
    def getTrainLoader(args, device):
        train_sets = imagenet.getTrainData(args)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loaders

    @staticmethod
    def getDataLoaders(args, device, shuffle_test=False):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        datasets = {}
        datasets['val'] = ImageNet(root=args.data_dir, split='val', transform=standard_transform())
        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = DataLoader(dataset, 
                                batch_size=args.eval_batch_size, 
                                shuffle=shuffle_test, 
                                sampler=None,
                                **kwargs)
        return tv_loaders
    
    @staticmethod
    def getTestLoader(args, device):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        test_dataset = ImageNet(root=args.data_dir, split='val', transform=transform_for_augmentation())
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=True,
                                 sampler=None,
                                 **kwargs)
        return test_loader
    

class imagenetv2(imagenet):
    def __init__(self):
        super(imagenetv2, self).__init__()

    @staticmethod
    def getTestData(args, use_transforms=False):
        transforms = te_transforms if use_transforms else None
        dataset = ImageNetV2Dataset(location=args.data_dir, transform=transforms)
        return dataset 

    @staticmethod
    def getDataLoaders(args, device, shuffle_test=False, split_tv=[]):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        datasets = {}
        datasets['id_val'] = ImageNet(root=args.data_dir, split='val', transform=standard_transform())
        datasets['val'] = ImageNetV2Dataset(location=args.data_dir, transform=standard_transform())
        tv_loaders = {}
        for split, dataset in datasets.items():
            if split in split_tv:
                train_size = int(0.2 * len(dataset))
                val_size = len(dataset) - train_size
                train_subset, val_subset = random_split(dataset, [train_size, val_size])
                
                tv_loaders[f'{split}_train'] = DataLoader(train_subset, batch_size=args.eval_batch_size, shuffle=True, **kwargs)
                tv_loaders[f'{split}_val'] = DataLoader(val_subset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)
            else:
                tv_loaders[split] = DataLoader(dataset, 
                                    batch_size=args.eval_batch_size, 
                                    shuffle=shuffle_test, 
                                    sampler=None,
                                    **kwargs)
        return tv_loaders
    
    @staticmethod
    def getTestLoader(args, device):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        
        test_dataset = ImageNetV2Dataset(location=args.data_dir, transform=transform_for_augmentation())
        test_loader = DataLoader(test_dataset, 
                                batch_size=args.eval_batch_size, 
                                shuffle=True, 
                                sampler=None,
                                **kwargs)
        return test_loader


class imagenet_r(imagenet):
    def __init__(self):
        super(imagenet_r, self).__init__()

    @staticmethod
    def getTestData(args, use_transforms=False):
        transforms = te_transforms if use_transforms else None
        dataset = ImageFolder(root=args.data_dir+'/imagenet-r', transform=transforms)
        return dataset

    @staticmethod
    def getDataLoaders(args, device, shuffle_test=False, split=False, split_tv=[]):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        datasets = {}
        datasets['id_val'] = ImageNet(root=args.data_dir, split='val', transform=standard_transform())
        datasets['val'] = ImageFolder(root=args.data_dir+'/imagenet-r', transform=standard_transform())
        tv_loaders = {}
        for split, dataset in datasets.items():
            if split in split_tv:
                train_size = int(0.2 * len(dataset))
                val_size = len(dataset) - train_size
                train_subset, val_subset = random_split(dataset, [train_size, val_size])
                
                tv_loaders[f'{split}_train'] = DataLoader(train_subset, batch_size=args.eval_batch_size, shuffle=True, **kwargs)
                tv_loaders[f'{split}_val'] = DataLoader(val_subset, batch_size=args.eval_batch_size, shuffle=False, **kwargs)
            else:
                tv_loaders[split] = DataLoader(dataset, 
                                    batch_size=args.eval_batch_size, 
                                    shuffle=shuffle_test, 
                                    sampler=None,
                                    **kwargs)
        return tv_loaders

    @staticmethod
    def project_logits(logits, device):
        return project_logits(logits, IMAGENET_R_CLASS_SUBLIST_MASK, device)
    

    @staticmethod
    def getTestLoader(args, device):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        test_dataset = ImageFolder(root=args.data_dir+'/imagenet-r', transform=transform_for_augmentation())
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=True,
                                 sampler=None,
                                 **kwargs)
        return test_loader