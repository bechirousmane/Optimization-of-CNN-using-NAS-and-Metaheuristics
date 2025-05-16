import random
import torch.nn as nn
from search_spaces.searchSpaceConfig import Config

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

def is_valid_architecture(layers:list)->bool:
    """
        Check if the architecture is valide
        Args :
           layers : list of dict
        Return : bool 
    """
    if not layers or len(layers) < Config.MIN_LAYERS or len(layers) > Config.MAX_LAYERS:
        return False
    if any(layer is None for layer in layers):
        return False
    # First layer will be Convolution layer
    if layers[0]["type"] != "Conv":
        return False
    # Last layer will be fully connected layer and at least one FC required
    last_fc_index = None
    for i, layer in enumerate(layers):
        if layer["type"] == "FC":
            last_fc_index = i
            break
    if last_fc_index is None:
        return False
    
    # All layers from last_fc_index must be FC
    for l in layers[last_fc_index:]:
        if l["type"] != "FC":
            return False
    # Banned Pool or FC in first layer
    if layers[0]["type"] in ["Pool", "FC"]:
        return False
    
    # No Pool followed by Pool
    for i in range(len(layers) - 1):
        if layers[i]["type"] == "Pool" and layers[i+1]["type"] == "Pool":
            return False
    return True

def build_torch_network(arch:list, input_shape=(3, 32, 32), num_classes=10):
    """
    Takes a bit string and generates a corresponding nn.Sequential.
    Args :
        input_shape: tuple, (channels, height, width)
        num_classes: int, Number of outputs from the last FC
    Return : nn.Sequential
    """
    modules = []
    
    # Initialize dimensions
    current_channels = input_shape[0]
    current_height = input_shape[1]
    current_width = input_shape[2]
    
    flatten_needed = False
    in_features = None
    
    for layer in arch:
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
