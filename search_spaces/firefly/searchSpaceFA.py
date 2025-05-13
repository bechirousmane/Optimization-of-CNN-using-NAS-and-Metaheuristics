import numpy as np
from ..searchSpaceConfig import Config
from ..utils import *

# Mapping layers type
LAYER_TYPE = {
    'None' : 0,
    'Conv': 1,
    'Pool': 2,
    'FC': 3
}

LAYER_SIZE = 4

def architecture_to_vector(arch:list[dict])->np.array : 
    """
        Convert the architecture ~~arch~~ to numpy array.
        Params : 
            acrh : list, the list that contains the architecture layers.
        Return : np.array
            Encoding the architecture as numpy array.
    """
    vector = []
    for layer in arch :
        v = [LAYER_TYPE[layer["type"]]]
        if layer["type"] == "Conv" :
            v.append(Config.FILTERS_MAP.index(layer["filters"]))
            v.append(Config.KERNEL_MAP.index(layer["kernel"]))
            v.append(Config.STRIDE_MAP.index(layer["stride"]))

        elif layer["type"] == "Pool" : 
            v.append(Config.KERNEL_MAP.index(layer["kernel"]))
            v.append(Config.STRIDE_MAP.index(layer["stride"]))
            v.append(0)
        elif layer["type"] == "FC" : 
            v.append(Config.FC_SIZES.index(layer["size"]))
            v.append(Config.ACTIVATION_FUNCTIONS.index(layer["activation"]))
            v.append(0)
        vector.extend(v)

    for _ in range(Config.MAX_LAYERS*LAYER_SIZE - len(vector)) : 
        vector.append(0)
    return np.array(vector)

def vector_to_achitecture(vector:np.array)->list :
    """
    Convert the vector ~~vector~~ to architecture as list of dictionnary.
    Params :
        vector : np.array, 
    Return: list
        The architecture of CNN as list of dictionnary
    """
    layers = vector.reshape((Config.MAX_LAYERS,LAYER_SIZE))
    arch = []
    for l in layers :
        layer = {}
        layer_type = l[0]
        if layer_type == 1 : # Conv
            layer["type"] = "Conv"
            layer["filters"] = Config.FILTERS_MAP[l[1]]
            layer["kernel"] = Config.KERNEL_MAP[l[2]]
            layer["stride"] = Config.STRIDE_MAP[l[3]]
        
        elif layer_type == 2 : # Pool
            layer["type"] = "Pool"
            layer["kernel"] = Config.KERNEL_MAP[l[1]]
            layer["stride"] = Config.STRIDE_MAP[l[2]]
        elif layer_type == 3  : # FC
            layer["type"] = "FC"
            layer["size"] = Config.FC_SIZES[l[1]]
            layer["activation"] = Config.ACTIVATION_FUNCTIONS[l[2]]
        else : 
            continue
        arch.append(layer)
    return arch


if __name__=="__main__" :
    arch = generate_valid_architecture()
    vector = architecture_to_vector(arch)
    print(arch)
    print("\n")
    print(vector)
    print("\n")
    print(vector_to_achitecture(vector))