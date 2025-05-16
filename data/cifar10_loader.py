import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar10(n_train=50000, n_test=10000, batch_size=64, num_workers=2, shuffle=True):
    """
        Loads the CIFAR10 dataset and returns the DataLoaders for training and testing.

        Args:
            n_train (int): Number of training samples to use (max 50,000)
            n_test (int): Number of test samples to use (max 10,000)
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of workers for DataLoaders
            shuffle (bool): If True, shuffles the training data

        Returns:
            tuple: (train_loader, test_loader) - DataLoaders for training and testing
    """
    # Limit Checking
    n_train = min(n_train, 50000)  #CIFAR10 has 50,000 training images
    n_test = min(n_test, 10000)    #CIFAR10 has 10000 test images
    
    # Defining transformations to normalize images
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Data augmentation: random cropping
        transforms.RandomHorizontalFlip(),     # Data augmentation: random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Mean and standard deviation for CIFAR10
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  
    ])
    
    # Downloading and preparing training data
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Downsample the training set if necessary
    if n_train < len(train_dataset):
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=transform_test
    )
    
    if n_test < len(test_dataset):
        indices = np.random.choice(len(test_dataset), n_test, replace=False)
        test_dataset = Subset(test_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers
    )
    
    print(f"CIFAR10 dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Data shape: {next(iter(train_loader))[0].shape}")  # [batch_size, channels, height, width]
    print(f"Classes: {train_dataset.dataset.classes if hasattr(train_dataset, 'dataset') else 'Not available for Subset'}")
    
    return train_loader, test_loader
