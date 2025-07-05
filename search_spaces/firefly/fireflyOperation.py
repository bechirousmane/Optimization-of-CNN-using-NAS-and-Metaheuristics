import numpy as np
from search_spaces.firefly.searchSpaceFA import *
from search_spaces.searchSpaceConfig import *
from search_spaces.utils import *

def euclideanDistance(vector1:np.array, vector2:np.array)->float :
    """
    Compute the euclidean distance between ~~vector1~~ and ~~vector2~~.
    Args :
        vector1 : np.array,
        vector2 : np.array
    Return :
        float, the euclidean distance between vector1 and vector2
    """
    return np.linalg.norm(vector1-vector2,ord=2)

def attractivenessFirfly(beta0:float, gamma:float, distance:float) -> float :
    """
        calculates the distance between two fireflies based on their distance
        Args :
            beta0 : float, the attractiveness at distance zero,
            gamma : float, the light absorption coefficient
            distance : float, the distance between the two fireflies
        Return : float,
            the attractiness value
    """
    return beta0*np.exp(-gamma*distance**2)

def moveFirefly(alpha:float, beta0:float, gamma:float, firefly:np.array, brightest_firefly:np.array)->np.array :
    """
        Move a firefly towards the brightest firefly.
        Args :
            alpha : float, control parameter
            beta0 : float, the attractiveness at distance zero,
            gamma : float, the lighr absorption coefficient
            firefly : np.array, the firefly to move
            brightest_firefly : np.array, the brightest firefly (best architecture)
        Return : np.array
            new architecture as numpy array
    """
    distance = euclideanDistance(firefly, brightest_firefly)
    attractiveness = attractivenessFirfly(beta0, gamma, distance)
    rand = np.random.uniform(0, 1, size=firefly.shape)
    newFirefly = firefly + attractiveness * (brightest_firefly - firefly) + alpha * (rand - 0.5)
    # Make sure the new firefly has valid integer values
    newFirefly = randomRounding(newFirefly)
    
    # Make sure the firefly is within the valid architecture space
    newFirefly = validateFirefly(newFirefly)
    
    return newFirefly

def randomRounding(vector) :
    return np.ceil(vector).astype(int) if random.random() < 1/2 else np.floor(vector).astype(int)

def validateFirefly(firefly:np.array)->np.array:
    """
    Ensure that the firefly represents a valid architecture in the search space.
    Args:
        firefly: np.array, the firefly vector to validate
    Return:
        np.array, a validated firefly vector
    """
    layers = firefly.reshape((Config.MAX_LAYERS, LAYER_SIZE))
    
    for i in range(Config.MAX_LAYERS):
        # Layer type must be between 0-3
        layers[i, 0] = max(0, min(3, layers[i, 0]))
        
        if layers[i, 0] == 1:  # Conv layer
            layers[i, 1] = max(0, min(len(Config.FILTERS_MAP) - 1, layers[i, 1]))
            layers[i, 2] = max(0, min(len(Config.KERNEL_MAP) - 1, layers[i, 2]))
            layers[i, 3] = max(0, min(len(Config.STRIDE_MAP) - 1, layers[i, 3]))
            
        elif layers[i, 0] == 2:  # Pool layer
            layers[i, 1] = max(0, min(len(Config.KERNEL_MAP) - 1, layers[i, 1]))
            layers[i, 2] = max(0, min(len(Config.STRIDE_MAP) - 1, layers[i, 2]))
            layers[i, 3] = 0
            
        elif layers[i, 0] == 3:  # FC layer
            layers[i, 1] = max(0, min(len(Config.FC_SIZES) - 1, layers[i, 1]))
            layers[i, 2] = max(0, min(len(Config.ACTIVATION_FUNCTIONS) - 1, layers[i, 2]))
            layers[i, 3] = 0
            
        else:  # None layer
            layers[i, :] = 0  # Reset all parameters
    
    # Flatten back to 1D array
    return layers.flatten()

def initializeFireflyPopulation(population_size:int)->list:
    """
    Initialize a population of fireflies with random valid architectures.
    Args:
        population_size: int, the number of fireflies to generate
    Return:
        list of list of dict, the population of fireflies
    """
    fireflies = []
    for _ in range(population_size):
        valid_arch = generate_valid_architecture()
        fireflies.append(valid_arch)
    return fireflies


