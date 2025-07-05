import torch
from torch.utils.data import Dataset


class LabelRemappingDataset(Dataset):
    """Dataset wrapper qui remappe les labels pour qu'ils soient consécutifs"""
    def __init__(self, dataset, selected_classes):
        self.dataset = dataset
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(selected_classes))}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Remapper le label
        new_label = self.label_mapping[label]
        return image, new_label

