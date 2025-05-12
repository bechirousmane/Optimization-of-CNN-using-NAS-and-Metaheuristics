import random
import torch.nn as nn
from searchSpaceConfig import Config
from searchSpaceGA import *


def generate_valid_architecture()->list:
    """
        Randomly generates a valid architecture
        Return : list of dict
    """
    n = random.randint(Config.MIN_LAYERS, Config.MAX_LAYERS)
    
    n_fc = random.randint(1, min(Config.MAX_LAYERS//2, n-1)) 
    n_main = n - n_fc

    layers = []

    # First layer must be a convolutional layer
    layer = {
        "type": "Conv",
        "filters": random.choice(Config.FILTERS_MAP),
        "kernel": random.choice(Config.KERNEL_MAP),
        "stride": random.choice(Config.STRIDE_MAP)
    }

    layers.append(layer)

    # Convolution and Pooling layers
    prev_type = "Conv"
    for i in range(1, n_main):
        possible_types = ["Conv", "Pool"]
        if prev_type == "Pool":
            possible_types.remove("Pool")
        t = random.choice(possible_types)
        if t == "Conv":
            layer = {
                "type": "Conv",
                "filters": random.choice(Config.FILTERS_MAP),
                "kernel": random.choice(Config.KERNEL_MAP),
                "stride": random.choice(Config.STRIDE_MAP)
            }
        elif t == "Pool":
            layer = {
                "type": "Pool",
                "kernel": random.choice(Config.KERNEL_MAP),
                "stride": random.choice(Config.STRIDE_MAP)
            }
        layers.append(layer)
        prev_type = t

    # Fully connected layers
    for _ in range(n_fc):
        layer = {
            "type": "FC",
            "size": random.choice(Config.FC_SIZES),
            "activation" : random.choice(Config.ACTIVATION_FUNCTIONS)
        }
        layers.append(layer)

    return layers


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

def build_torch_network(arch:str, input_shape=(3, 32, 32), num_classes=10):
    """
    Takes a bit string and generates a corresponding nn.Sequential.
    Params :
        input_shape: tuple, (channels, height, width)
        num_classes: int, Number of outputs from the last FC
    Return : nn.Sequential
    """
    architecture = binary_to_architecture(arch)
    modules = []
    
    # Initialize dimensions
    current_channels = input_shape[0]
    current_height = input_shape[1]
    current_width = input_shape[2]
    
    flatten_needed = False
    in_features = None
    
    for layer in architecture:
        if layer['type'] == 'Conv':
            modules.append(nn.Conv2d(
                current_channels, layer['filters'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                padding=Config.CONV_PADDING
            ))
            modules.append(nn.ReLU())
            
            # Update channels
            current_channels = layer['filters']
            
            # Calculate new spatial dimensions after convolution
            # Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
            padding = Config.CONV_PADDING
            kernel_size = layer['kernel']
            stride = layer['stride']
            
            current_height = int((current_height - kernel_size + 2*padding) / stride + 1)
            current_width = int((current_width - kernel_size + 2*padding) / stride + 1)
            
            flatten_needed = True
            
        elif layer['type'] == 'Pool':
            modules.append(nn.MaxPool2d(
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                padding=Config.POOL_PADDING
            ))
            
            # Update spatial dimensions after pooling
            padding = Config.POOL_PADDING
            kernel_size = layer['kernel']
            stride = layer['stride']
            
            current_height = int((current_height - kernel_size + 2*padding) / stride + 1)
            current_width = int((current_width - kernel_size + 2*padding) / stride + 1)
            
        elif layer['type'] == 'FC':
            if flatten_needed:
                modules.append(nn.Flatten())
                flatten_needed = False
                # Dynamic calculation of in_features based on current dimensions
                in_features = current_channels * current_height * current_width
            
            out_features = layer['size']
            modules.append(nn.Linear(in_features, out_features,bias=True))
            modules.append(layer['activation']())
            in_features = out_features
    
    # Add final layer
    if in_features is None:
        # If no FC layer was in the architecture, we need to flatten and calculate in_features
        modules.append(nn.Flatten())
        in_features = current_channels * current_height * current_width
    modules.append(nn.Linear(in_features, num_classes, bias=True))
    modules.append(nn.ReLU())
    
    return nn.Sequential(*modules)



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
    net = build_torch_network(mutant, input_shape=(3,32,32), num_classes=10)
    print(net)
    #print(net)
    print(is_valid_architecture(binary_to_architecture(mutant)))


