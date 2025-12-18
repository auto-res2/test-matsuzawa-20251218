import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from omegaconf import DictConfig
from pathlib import Path
import numpy as np

class CIFAR10Dataset:
    """CIFAR-10 dataset with preprocessing."""
    
    def __init__(self, data_root="./data/cifar10", split="train", normalize=True, augmentation=False):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Base transforms
        transforms_list = []
        
        if augmentation and split == "train":
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
        
        if normalize:
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                ),
            ])
        else:
            transforms_list.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transforms_list)
        
        # Load dataset
        is_train = split == "train"
        self.dataset = datasets.CIFAR10(
            root=str(self.data_root),
            train=is_train,
            download=True,
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class CIFAR100Dataset:
    """CIFAR-100 dataset with preprocessing."""
    
    def __init__(self, data_root="./data/cifar100", split="train", normalize=True, augmentation=False):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Base transforms
        transforms_list = []
        
        if augmentation and split == "train":
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
        
        if normalize:
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5070, 0.4865, 0.4409),
                    std=(0.2009, 0.1984, 0.2023)
                ),
            ])
        else:
            transforms_list.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transforms_list)
        
        # Load dataset
        is_train = split == "train"
        self.dataset = datasets.CIFAR100(
            root=str(self.data_root),
            train=is_train,
            download=True,
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def build_dataset(dataset_cfg: DictConfig, model_type: str = "resnet"):
    """Build train/val/test datasets from configuration."""
    
    dataset_name = dataset_cfg.get("name", "CIFAR-10").upper()
    data_root = dataset_cfg.get("data_root", "./data")
    normalize = dataset_cfg.get("preprocessing", {}).get("normalize", True)
    augmentation = dataset_cfg.get("preprocessing", {}).get("augmentation", True)
    
    if "CIFAR-10" in dataset_name:
        full_train = CIFAR10Dataset(
            data_root=data_root.replace("cifar100", "cifar10"),
            split="train",
            normalize=normalize,
            augmentation=augmentation
        )
        full_test = CIFAR10Dataset(
            data_root=data_root.replace("cifar100", "cifar10"),
            split="test",
            normalize=normalize,
            augmentation=False
        )
    elif "CIFAR-100" in dataset_name:
        full_train = CIFAR100Dataset(
            data_root=data_root.replace("cifar10", "cifar100"),
            split="train",
            normalize=normalize,
            augmentation=augmentation
        )
        full_test = CIFAR100Dataset(
            data_root=data_root.replace("cifar10", "cifar100"),
            split="test",
            normalize=normalize,
            augmentation=False
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split train into train/val
    train_split = dataset_cfg.get("train_split", 0.9)
    num_train = len(full_train)
    num_train_samples = int(train_split * num_train)
    
    train_indices = list(range(num_train_samples))
    val_indices = list(range(num_train_samples, num_train))
    
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)
    test_dataset = full_test
    
    # Post-init assertions
    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"
    
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=128, num_workers=4):
    """Create data loaders."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Post-init assertions
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"
    
    return train_loader, val_loader, test_loader
