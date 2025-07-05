import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from data.data_remapping import LabelRemappingDataset

def load_mnist(n_train=60000, n_test=10000, n_class=10, batch_size=64, num_workers=2, shuffle=True):
    """
    Loads the MNIST dataset and returns the DataLoaders for training and testing.
    Args:
        n_class (int) : Number of class to use. min 2 max 10
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
    n_class = min(max(n_class, 2), 10)
    
    # Defining transformations to normalize images
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor() 
    ])
    
    # Downloading and preparing training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    selected_classes = list(range(10))
    # Filter by class if n_class < 10
    if n_class < 10:
        # Randomly select n_class classes from 0-9
        selected_classes = np.random.choice(10, n_class, replace=False)
        # Get indices for the selected classes
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
        train_dataset = Subset(train_dataset, train_indices)
        train_dataset = LabelRemappingDataset(train_dataset, selected_classes)
    
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
    
    # Filter by class if n_class < 10
    if n_class < 10:
        # Get indices for the selected classes
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]
        test_dataset = Subset(test_dataset, test_indices)
        test_dataset = LabelRemappingDataset(test_dataset, selected_classes)

    
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
    if n_class < 10:
        print(f"Using {n_class} randomly selected classes: {sorted(selected_classes)}")
    else:
        print(f"Using all 10 classes (0-9)")
    print(f"Data shape: {next(iter(train_loader))[0].shape}")  

    return train_loader, test_loader, selected_classes

