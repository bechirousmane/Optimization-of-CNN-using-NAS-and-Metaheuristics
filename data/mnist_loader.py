import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_mnist(n_train=60000, n_test=10000, batch_size=64, num_workers=2, shuffle=True):
    """
        Loads the MNIST dataset and returns the DataLoaders for training and testing.
        
        Args:
            n_train (int): Number of training samples to use (max 60,000)
            n_test (int): Number of test samples to use (max 10,000)
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of workers for DataLoaders
            shuffle (bool): If True, shuffles the training data

        Returns:
            tuple: (train_loader, test_loader) - DataLoaders for training and testing
    """
    # Limit Checking
    n_train = min(n_train, 60000)  
    n_test = min(n_test, 10000)   
    
    # Defining transformations to normalize images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation of the MNIST dataset
    ])
    
    # Downloading and preparing training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    # Downsample the training set if necessary
    if n_train < len(train_dataset):
        # Generating random indices for subsampling
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # Downloading and preparing test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    # Downsample the test set if necessary
    if n_test < len(test_dataset):
        # Generating random indices for subsampling
        indices = np.random.choice(len(test_dataset), n_test, replace=False)
        test_dataset = Subset(test_dataset, indices)
    
    # Creating DataLoaders
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
    
    print(f"MNIST dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Data shape: {next(iter(train_loader))[0].shape}")  # Displays the shape of the data: [batch_size, channels, height, width]
    
    return train_loader, test_loader
