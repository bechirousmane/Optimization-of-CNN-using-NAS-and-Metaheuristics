import torch.nn as nn
from search_spaces.searchSpaceConfig import Config

SIZE_INDIVIDUAL = 8

def encode_layer(layer:dict)->str:
    """
        Encoding the layer ~~layer~~ in binary string
        Args :
            layer : dictionary
        Return : str
    """
    if layer is None:
        return "00" + "000000"
    t = layer["type"]
    if t == "Conv":
        type_bits = "01"
        nb_filters = Config.FILTERS_MAP.index(layer['filters'])
        kernel_size = Config.KERNEL_MAP.index(layer['kernel'])
        stride = Config.STRIDE_MAP.index(layer['stride'])
        params = f"{nb_filters:02b}{kernel_size:02b}{stride:02b}"
    elif t == "Pool":
        type_bits = "10"
        kernel_size = Config.KERNEL_MAP.index(layer['kernel'])
        stride = Config.STRIDE_MAP.index(layer['stride'])
        params = f"{kernel_size:02b}{stride:02b}00"
    elif t == "FC":
        type_bits = "11"
        layer_size = Config.FC_SIZES.index(layer['size'])
        activation_function = Config.ACTIVATION_FUNCTIONS.index(layer['activation'])
        params = f"{layer_size:04b}{activation_function:02b}"
    return type_bits + params

def decode_layer(bits:str)->dict:
    """
        Decoding the bit sequence ~~bits~~ in dictionary.
        Args :
            bits : str
        Return : dict
    """
    if len(bits) != SIZE_INDIVIDUAL:
        return None
    layer_type = bits[:2]
    params = bits[2:]
    if layer_type == "00":
        return None
    elif layer_type == "01":  # Conv
        filters = Config.FILTERS_MAP[int(params[:2], 2)]
        kernel = Config.KERNEL_MAP[int(params[2:4], 2)]
        stride = Config.STRIDE_MAP[int(params[4:6], 2)]
        return {"type": "Conv", "filters": filters, "kernel": kernel, "stride": stride}
    elif layer_type == "10":
        kernel = Config.KERNEL_MAP[int(params[:2], 2)]
        stride = Config.STRIDE_MAP[int(params[2:4], 2)]
        return {"type": "Pool", "kernel": kernel, "stride": stride}
    elif layer_type == "11":
        size = Config.FC_SIZES[int(params[:4], 2)]
        activation = Config.ACTIVATION_FUNCTIONS[int(params[4:],2)]
        return {"type": "FC", "size": size, "activation":activation}
    return None

def architecture_to_binary(layers:list)->str:
    """
        Encoding the layers list ~~layers~~ into binary string.
        Args :
            layers : list of dict
        Return : 
            str
    """
    return ''.join(encode_layer(layer) for layer in layers)

def binary_to_architecture(binary_string):
    """
        Decode the bit sequence ~~binary_string~~ into a dictionary-like architecture
        Args :
            binary_string : str
        Return : list of dict
    """
    layers = []
    for i in range(0, len(binary_string), 8):
        layer_bits = binary_string[i:i+8]
        layer = decode_layer(layer_bits)
        if layer:
            layers.append(layer)
    return layers


