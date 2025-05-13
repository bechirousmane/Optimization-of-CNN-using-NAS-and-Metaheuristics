import unittest
import torch.nn as nn
from .searchSpaceFA import *
from .fireflyOperation import * 
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
    
    def test_complete_architecture_encoding_decoding(self):
        """Test encoding and decoding of a complete valid architecture"""
        original_arch = [
            {
                "type": "Conv",
                "filters": 32,
                "kernel": 3,
                "stride": 1
            },
            {
                "type": "Pool",
                "kernel": 3,
                "stride": 2
            },
            {
                "type": "Conv",
                "filters": 64,
                "kernel": 3,
                "stride": 1
            },
            {
                "type": "FC",
                "size": 128,
                "activation": nn.ReLU
            },
            {
                "type": "FC",
                "size": 64,
                "activation": nn.Tanh
            }
        ]
        
        # Check that this is a valid architecture first
        self.assertTrue(is_valid_architecture(original_arch))
        
        # Encode and decode
        vector = architecture_to_vector(original_arch)
        decoded_arch = vector_to_achitecture(vector)
        
        # Verify structure is preserved
        self.assertEqual(len(original_arch), len(decoded_arch))
        
        # Verify each layer's properties
        for i in range(len(original_arch)):
            self.assertEqual(original_arch[i]["type"], decoded_arch[i]["type"])
            
            if original_arch[i]["type"] == "Conv":
                self.assertEqual(original_arch[i]["filters"], decoded_arch[i]["filters"])
                self.assertEqual(original_arch[i]["kernel"], decoded_arch[i]["kernel"])
                self.assertEqual(original_arch[i]["stride"], decoded_arch[i]["stride"])
            elif original_arch[i]["type"] == "Pool":
                self.assertEqual(original_arch[i]["kernel"], decoded_arch[i]["kernel"])
                self.assertEqual(original_arch[i]["stride"], decoded_arch[i]["stride"])
            elif original_arch[i]["type"] == "FC":
                self.assertEqual(original_arch[i]["size"], decoded_arch[i]["size"])
                self.assertEqual(original_arch[i]["activation"], decoded_arch[i]["activation"])
    
    def test_euclidean_distance(self):
        """Test the euclidean distance calculation between two fireflies"""
        firefly1 = np.array([1, 2, 3, 4, 5])
        firefly2 = np.array([5, 4, 3, 2, 1])
        
        # Calculate expected distance
        expected_distance = np.sqrt(np.sum((firefly1 - firefly2)**2))
        
        calculated_distance = euclideanDistance(firefly1, firefly2)
        
        self.assertAlmostEqual(expected_distance, calculated_distance)
    
    def test_attractiveness_firefly(self):
        """Test the attractiveness calculation between two fireflies"""
        beta0 = 1.0  # Base attractiveness
        gamma = 0.5  # Light absorption coefficient
        distance = 2.0
        
        # Calculate expected attractiveness
        expected_attractiveness = beta0 * np.exp(-gamma * distance**2)
        
        calculated_attractiveness = attractivenessFirfly(beta0, gamma, distance)
        
        self.assertAlmostEqual(expected_attractiveness, calculated_attractiveness)
    
    def test_move_firefly(self):
        """Test that a firefly moves toward a brighter firefly"""
        # Create two fireflies very close, so they must have a strong attractiveness
        arch1 = [
            {
                "type": "Conv",
                "filters": 32,
                "kernel": 3,
                "stride": 1
            },
            {
                "type":"Pool",
                "kernel": 3,
                "stride": 2
            },
            {
                "type": "FC",
                "size": 8,
                "activation": nn.ReLU
            }            
        ]
        
        arch2 = [
            {
                "type": "Conv",
                "filters": 32,
                "kernel": 3,
                "stride": 2
            },
            {
                "type":"Pool",
                "kernel": 3,
                "stride": 2
            },
            {
                "type": "FC",
                "size": 16,
                "activation": nn.ELU
            }
        ]

        # Create two fireflies very distant, so they must have a low attractiveness
        arch3 = [
            {
                "type": "Conv",
                "filters": 64,
                "kernel": 5,
                "stride": 4
            },
            {
                "type":"Pool",
                "kernel": 7,
                "stride": 4
            },
            {
                "type": "FC",
                "size": 256,
                "activation": nn.ReLU
            }            
        ]
        
        arch4 = [
            {
                "type": "Conv",
                "filters": 32,
                "kernel": 9,
                "stride": 1
            },
            {
                "type":"Pool",
                "kernel": 5,
                "stride": 1
            },
            {
                "type": "FC",
                "size": 16,
                "activation": nn.Tanh
            }
        ]
        
        firefly1 = architecture_to_vector(arch1)
        brightest_firefly1 = architecture_to_vector(arch2)
        firefly2 = architecture_to_vector(arch3)
        brightest_firefly2 = architecture_to_vector(arch4)
        
        # Set parameters
        alpha = 0.5
        beta0 = 1.0
        gamma = 0.2
        
        # Move the firefly
        new_firefly1 = moveFirefly(alpha, beta0, gamma, firefly1, brightest_firefly1)
        new_firefly2 = moveFirefly(alpha, beta0, gamma, firefly2, brightest_firefly2)

        # Check that the new_firefly1 is not identical to the original 
        self.assertFalse(np.array_equal(firefly1, new_firefly1))
        self.assertLess(euclideanDistance(new_firefly1, brightest_firefly1), euclideanDistance(firefly1, brightest_firefly1))

        # Check that the new_firefly2 is identical to the original. This is explained by the fact that distant fireflies (architectures) 
        #  are less attracted to each other. Moreover, since the search space is discrete and the movement is so minimal, 
        # it is considered negligible in our search space.
        self.assertTrue(np.array_equal(firefly2, new_firefly2))
       
        # Convert to architecture and check if it's valid
        new_arch = vector_to_achitecture(new_firefly1)
        self.assertTrue(is_valid_architecture(new_arch))

    
    def test_validate_firefly(self):
        """Test that validateFirefly makes an invalid firefly valid"""
        # Create an invalid firefly vector 
        invalid_arch = [
            {
                "type": "Pool",  
                "kernel": 3,
                "stride": 1
            },
            {
                "type": "Pool",  
                "kernel": 5,
                "stride": 2
            },
            {
                "type": "Conv",  
                "filters": 64,
                "kernel": 3,
                "stride": 1
            },
            {
                "type": "FC",
                "size": 128,
                "activation": nn.ReLU
            }
        ]
        
        invalid_vector = architecture_to_vector(invalid_arch)
        
        # Validate the firefly
        valid_vector = validateFirefly(invalid_vector)
        
        # Convert back to architecture and check if it's valid
        valid_arch = vector_to_achitecture(valid_vector)
        self.assertTrue(is_valid_architecture(valid_arch))
        
        
    def test_initialize_firefly_population(self):
        """Test generating a population of valid firefly architectures"""
        population_size = 5
        population = initializeFireflyPopulation(population_size)
        
        # Check population size
        self.assertEqual(len(population), population_size)
        
        # Check that each firefly represents a valid architecture
        for firefly in population:
            arch = vector_to_achitecture(firefly)
            self.assertTrue(is_valid_architecture(arch))
    
    
if __name__ == '__main__':
    unittest.main()