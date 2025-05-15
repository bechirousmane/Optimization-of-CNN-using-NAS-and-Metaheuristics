# Exemple d'utilisation
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .random_search import RandomSearch
from .genetic_search import GeneticSearch
from .firefly_search import FireFlySearch

# Préparer vos données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Créer et lancer la recherche aléatoire
search = FireFlySearch(
    alpha=0.5,
    beta0=2,
    gamma=0.7,
    population_size=30,
    iterations=10,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=2,
    use_gpu=torch.cuda.is_available()
)

best_arch, best_fitness, history = search.run_search()
print(f"Meilleure architecture trouvée: {best_arch}")
print(f"Meilleur score de fitness: {best_fitness}")
print(f"Nombre d'appele à la fonction objective : {search.count_eval}")