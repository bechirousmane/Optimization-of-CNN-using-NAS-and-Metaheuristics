import torch.nn as nn

class Config :
    FILTERS_MAP = [1, 8, 16, 32]
    KERNEL_MAP = [3, 5, 7, 9]
    STRIDE_MAP = [1, 2, 3, 4]
    CONV_PADDING = 1
    POOL_PADDING = 1
    ACTIVATION_FUNCTIONS = [nn.ReLU, nn.ELU, nn.Sigmoid, nn.Tanh]
    FC_SIZES = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 512]
    MAX_LAYERS = 15
    MIN_LAYERS = 2
    CROMOSOME_SIZE = 8
