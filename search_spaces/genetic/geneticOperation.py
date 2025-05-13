import random
import math
import torch.nn as nn
from ..searchSpaceConfig import Config
from .searchSpaceGA import *
from ..utils import *

def elitisteSelection(population:list, fitness, nbr:int, **kwarg)->list :
    """
        Return the ~~nbr~~ individuals with the best fitness.
        Params :
            population : list, list of architectures.
            fitness : function, evaluation function.
            kwarg : the fitness function arguments
            nbr : int, the number of individuals to be returned.
        Return : list
    """
    # If requested number is greater than population size, return entire population
    if nbr >= len(population):
        return population
    
    # Calculate fitness for each individual
    # We use a list comprehension to create (individual, fitness_score) tuples
    fitness_scores = [(individual, fitness(individual, **kwarg)) for individual in population]
    
    # Sort the list based on fitness scores in descending order
    sorted_population = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
    
    # Return the top 'nbr' individuals (without their fitness scores)
    return [individual for individual, _ in sorted_population[:nbr]]

def polynomialRankSelection(population: list, fitness, nbr: int, selective_pressure=1.5, **kwargs) -> list:
    """
        Perform polynomial rank-based selection.
        
        Params:
            population : list, list of architectures
            fitness : function, evaluation function
            nbr : int, number of individuals to be selected
            selective_pressure : float, the pressure of selection
            kwargs : additional arguments for fitness function
        
        Returns:
        list: Selected individuals based on polynomial rank-based selection
    """
    # Compute fitness for each individual
    fitness_scores = [(individual, fitness(individual, **kwargs)) for individual in population]
    
    # Sort individuals by fitness in descending order
    sorted_population = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
    
    n = len(population)
    
    # Calculate selection probabilities using polynomial ranking
    selection_probs = []
    for rank in range(1, n + 1):
        # Probability of selection decreases polynomially with rank
        prob = (2 - selective_pressure) / n + (2 * rank * (selective_pressure - 1)) / (n * (n - 1))
        selection_probs.append(prob)
    
    # Normalize probabilities to ensure they sum to 1
    total_prob = sum(selection_probs)
    selection_probs = [p / total_prob for p in selection_probs]
    
    # Select individuals based on these probabilities
    selected = []
    for _ in range(nbr):
        # Use cumulative probability for selection
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(selection_probs):
            cumulative_prob += prob
            if r <= cumulative_prob:
                selected.append(sorted_population[i][0])
                break
    
    return selected

def probabilisticTournamentSelection(population: list, fitness, nbr: int, tournament_size = 3, tournament_prob=.75, **kwargs) -> list:
    """
        Perform probabilistic tournament selection.
        
        Params:
            population : list, list of architectures
            fitness : function, evaluation function
            nbr : int, number of individuals to be selected
            tournament_size : int, the tournament size
            tournament_prob: float
            kwargs : additional arguments for fitness function
        
        Returns:
            list: Selected individuals based on probabilistic tournament selection
    """
    
    # Compute fitness for each individual
    fitness_scores = [(individual, fitness(individual, **kwargs)) for individual in population]
    
    selected = []
    for _ in range(nbr):
        # Randomly select tournament_size individuals
        tournament_candidates_idx = random.sample(list(range(len(population))), tournament_size)
        
        # Sort tournament candidates by fitness
        sorted_tournament_idx = sorted(tournament_candidates_idx, key=lambda x: fitness_scores[x][1], reverse=True)
        sorted_tournament = [population[i] for i in sorted_tournament_idx]
        # Probabilistic selection within the tournament
        r = random.random()
        if r < tournament_prob:
            # Select the best individual with higher probability
            selected.append(sorted_tournament[0])
        else:
            # Randomly select from the tournament with lower probability
            selected.append(random.choice(sorted_tournament))
    
    return selected



def onePointCrossover(parent1:str, parent2:str)-> str:
    """
        Makes a one-point crossover of two parents.
        Params : 
            parent1 : str, binary string
            parent2 : str, binary string
        Return : str, binary string
    """
    cut_off_point1 = random.choice([i for i in range(len(parent1)) if i % Config.CROMOSOME_SIZE==0])
    cut_off_point2 = random.choice([i for i in range(len(parent2)) if i % Config.CROMOSOME_SIZE==0])
    child1 = parent1[:cut_off_point1] + parent2[cut_off_point2:]
    child2 = parent2[:cut_off_point2] + parent1[cut_off_point1:]
    return child1 if is_valid_architecture(child1) else child2

def mutate(architecture:str, mutation_rate=0.1)->str:
    """
        Make a mutation with a probability of ~~mutation_rate~~ on the architecture ~~architecture~~
        Params :
            architecture : str, binary string
            mutation_rate : float, probability bitween 0 and 1
        Return : str, binary string
    """
    idx = list(range(len(architecture)))
    nb_bits_to_mutate = min(len(architecture)//3, random.randint(0, len(architecture)))
    idx_bits_to_mutate = random.sample(idx, nb_bits_to_mutate)
    for i in idx_bits_to_mutate:
        if random.random() < mutation_rate and i%8 != 0:
            architecture = architecture[:i] + ('0' if architecture[i]=='1' else '1') + architecture[i+1:]
    return architecture




if __name__=="__main__" : 
    arch1 = generate_valid_architecture()
    arch2 = generate_valid_architecture()
    child = onePointCrossover(architecture_to_binary(arch1), architecture_to_binary(arch2))
    mutant = mutate(child)

    print("Parent 1 :\n", arch1,"\n")
    print("Parent 2 :\n", arch2,"\n")
    print("Child after crossover :\n", f"{binary_to_architecture(child)}\n")
    print("Child after mutation :\n", f"{binary_to_architecture(mutant)}\n")
    print("Binary string :\n", f"{mutant}\n")
    net = build_torch_network(binary_to_architecture(mutant), input_shape=(3,32,32), num_classes=10)
    print(net)
    #print(net)
    print(is_valid_architecture(binary_to_architecture(mutant)))


