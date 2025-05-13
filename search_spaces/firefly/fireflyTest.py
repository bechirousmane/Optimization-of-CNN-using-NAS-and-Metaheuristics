import unittest
import torch.nn as nn
from .searchSpaceFA import *
from ..utils import * 
from ..searchSpaceConfig import *

class TestFireflySearchSpacesFunction(unittest.TestCase) :
    def test_encode_decode_layer_conv(self):
        """Test encoding and decoding a Conv layer"""
        conv_layer = {
            "type": "Conv",
            "filters": 32,
            "kernel": 3,
            "stride": 1
        }
        vector = architecture_to_vector([conv_layer])
        self.assertEqual(len(vector), Config.MAX_LAYERS*LAYER_SIZE)
        self.assertEqual(vector[0], 1)  # Conv type
        decoded = vector_to_achitecture(vector)
        decoded = decoded[0]
        self.assertEqual(decoded["type"], conv_layer["type"])
        self.assertEqual(decoded["filters"], conv_layer["filters"])
        self.assertEqual(decoded["kernel"], conv_layer["kernel"])
        self.assertEqual(decoded["stride"], conv_layer["stride"])
    
    def test_encode_decode_layer_pool(self):
        """Test encoding and decoding a Pool layer"""
        pool_layer = {
            "type": "Pool",
            "kernel": 5,
            "stride": 2
        }
        vector = architecture_to_vector([pool_layer])
        self.assertEqual(len(vector), Config.MAX_LAYERS*LAYER_SIZE)
        self.assertEqual(vector[0], 2)  # Pool type
        decoded = vector_to_achitecture(vector)
        decoded = decoded[0]
        self.assertEqual(decoded["type"], pool_layer["type"])
        self.assertEqual(decoded["kernel"], pool_layer["kernel"])
        self.assertEqual(decoded["stride"], pool_layer["stride"])
    
    def test_encode_decode_layer_fc(self):
        """Test encoding and decoding a FC layer"""
        fc_layer = {
            "type": "FC",
            "size": 64,
            "activation": nn.ReLU
        }
        vector = architecture_to_vector([fc_layer])
        self.assertEqual(len(vector), Config.MAX_LAYERS*LAYER_SIZE)
        self.assertEqual(vector[0], 3)  # FC type
        decoded = vector_to_achitecture(vector)
        decoded = decoded[0]
        self.assertEqual(decoded["type"], fc_layer["type"])
        self.assertEqual(decoded["size"], fc_layer["size"])
        self.assertEqual(decoded["activation"], fc_layer["activation"])
    

if __name__ == '__main__':
    unittest.main()