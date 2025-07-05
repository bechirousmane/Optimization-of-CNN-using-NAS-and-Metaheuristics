import asyncio
import torch
import torch.nn as nn
import numpy as np
from search_spaces.utils import build_torch_network
from train.trainer import ModelTrainer
from ressource.ressource_manager import ResourceManager
from data.cifar10_loader import  load_cifar10

random_model = build_torch_network([{'type': 'Conv', 'filters': 128, 'kernel': 5, 'stride': 2}, {'type': 'Conv', 'filters': 256, 'kernel': 5, 'stride': 1}, {'type': 'Pool', 'kernel': 2, 'stride': 2}, {'type': 'Conv', 'filters': 64, 'kernel': 5, 'stride': 1}, {'type': 'FC', 'size': 224, 'activation': nn.ELU}, {'type': 'FC', 'size': 80, 'activation':nn.Sigmoid}])

gentic_model = build_torch_network([{'type': 'Conv', 'filters': 64, 'kernel': 3, 'stride': 1}, {'type': 'Conv', 'filters': 256, 'kernel': 3, 'stride': 2}, {'type': 'Conv', 'filters': 128, 'kernel': 3, 'stride': 2}, {'type': 'Conv', 'filters': 128, 'kernel': 3, 'stride': 1}, {'type': 'Conv', 'filters': 256, 'kernel': 3, 'stride': 1}, {'type': 'FC', 'size': 384, 'activation': nn.ReLU}])

firefly_model = build_torch_network( [{'type': 'Conv', 'filters': 32, 'kernel': 2, 'stride': 1}, {'type': 'Conv', 'filters': 32, 'kernel': 5, 'stride': 1}, {'type': 'Pool', 'kernel': 3, 'stride': 1}, {'type': 'Conv', 'filters': 128, 'kernel': 2, 'stride': 2}, {'type': 'Conv', 'filters': 128, 'kernel': 3, 'stride': 3}, {'type': 'Conv', 'filters': 64, 'kernel': 5, 'stride': 1}, {'type': 'Pool', 'kernel': 3, 'stride': 2}])

print(f"random :\n{random_model}\n\ngenetic : \n{gentic_model}\n\nfirefly : {firefly_model}")

train, test, _ = load_cifar10(batch_size=128)
 
random_trainer = ModelTrainer(
    model=random_model,
    device="cpu",
    lr=1e-3,
    epochs=100,
    train_loader=train,
    test_loader=test,
    optimizer="AdamW"
)

genetic_trainer = ModelTrainer(
    model=gentic_model,
    device="cpu",
    lr=1e-3,
    epochs=100,
    train_loader=train,
    test_loader=test,
    optimizer="AdamW"
)

firefly_trainer = ModelTrainer(
    model=firefly_model,
    device="cpu",
    lr=1e-3,
    epochs=100,
    train_loader=train,
    test_loader=test,
    optimizer="AdamW"
)

print(f"{10 * "-"} Random model trainning...{10 * "-"}")
random_train_loss_histoty, random_train_precision_history = random_trainer.train(verbose=True)
print(f"{10 * "-"}Random model precision on test set :{random_trainer.test()} {10 * "-"}")

print(f"{10 * "-"} genetic model trainning...{10 * "-"}")
genetic_train_loss_histoty, genetic_train_precision_history = genetic_trainer.train(verbose=True)
print(f"genetic model precision on test set :{genetic_trainer.test()} ")

print(f"{10 * "-"} firefly model trainning...{10 * "-"}")
firefly_train_loss_histoty, firefly_train_precision_history = firefly_trainer.train(verbose=True)
print(f"firefly model precision on test set :{firefly_trainer.test()} ")

