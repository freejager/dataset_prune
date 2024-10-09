import time
import torch
import random
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import CustomSubset
########################################################################################################################
# Load Data
########################################################################################################################
    
def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55 = load_dataset(args)
    return train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10':
        train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55 = load_cifar10(args)
    elif args.dataset == 'cifar100':
        train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55 = load_cifar100(args)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55
def random_prune_dataset(dataset, rate):

    total_size = len(dataset)
    num_to_keep = int(total_size * rate)

    result = torch.randperm(total_size)

    indices, _ = torch.sort(result[:num_to_keep])
    other_indices, _ = torch.sort(result[num_to_keep:])

    coreset = CustomSubset(dataset, indices)
    otherset = CustomSubset(dataset, other_indices)
    return coreset

def random_split(dataset, num_splits=5):
    total_size = len(dataset)
    part_size = total_size // num_splits
    indices = torch.randperm(total_size)
    
    subsets = []
    complements = []
    start = 0
    for i in range(num_splits):
        end = start + part_size if i < num_splits - 1 else total_size
        subset_indices = indices[start:end]
        subset_indices, _ = torch.sort(subset_indices)
        
        # Create the subset
        subsets.append(CustomSubset(dataset, subset_indices))
        
        # Create the complement of the subset
        complement_indices = torch.cat((indices[:start], indices[end:]))
        complement_indices, _ = torch.sort(complement_indices)
        complements.append(CustomSubset(dataset, complement_indices))
        
        start = end

    return subsets, complements


def load_cifar10(args):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    train_data.target = train_data.targets

    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
        
    subsets, complements = random_split(train_data, 5)
    dataset1,dataset2,dataset3,dataset4,dataset5=subsets
    dataset11,dataset22,dataset33,dataset44,dataset55=complements

    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_1 = torch.utils.data.DataLoader(dataset1, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_2 = torch.utils.data.DataLoader(dataset2, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_3 = torch.utils.data.DataLoader(dataset3, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_4 = torch.utils.data.DataLoader(dataset4, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_5 = torch.utils.data.DataLoader(dataset5, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    train_loader_11 = torch.utils.data.DataLoader(dataset11, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_22 = torch.utils.data.DataLoader(dataset22, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_33 = torch.utils.data.DataLoader(dataset33, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_44 = torch.utils.data.DataLoader(dataset44, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_55 = torch.utils.data.DataLoader(dataset55, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)




    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    train_data.target = train_data.targets
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    subsets, complements = random_split(train_data, 5)
    dataset1,dataset2,dataset3,dataset4,dataset5=subsets
    dataset11,dataset22,dataset33,dataset44,dataset55=complements
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_1 = torch.utils.data.DataLoader(dataset1, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_2 = torch.utils.data.DataLoader(dataset2, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_3 = torch.utils.data.DataLoader(dataset3, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_4 = torch.utils.data.DataLoader(dataset4, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_5 = torch.utils.data.DataLoader(dataset5, args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    train_loader_11 = torch.utils.data.DataLoader(dataset11, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_22 = torch.utils.data.DataLoader(dataset22, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_33 = torch.utils.data.DataLoader(dataset33, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_44 = torch.utils.data.DataLoader(dataset44, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    train_loader_55 = torch.utils.data.DataLoader(dataset55, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)    
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_data, train_loader, test_loader, train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5, train_loader_11, train_loader_22, train_loader_33, train_loader_44, train_loader_55
