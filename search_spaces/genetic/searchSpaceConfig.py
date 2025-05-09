import torch.nn as nn

class Config :
    FILTERS_MAP = [16, 32, 64, 128]
    KERNEL_MAP = [3, 5, 7, 9]
    STRIDE_MAP = [1, 2, 3, 4]
    ACTIVATION_FUNCTIONS = [nn.ReLU, nn.ELU, nn.LeakyReLU, nn.Tanh]
    FC_SIZES = [16 * (2**i) for i in range(16)]
    MAX_LAYERS = 10
    MIN_LAYERS = 3
