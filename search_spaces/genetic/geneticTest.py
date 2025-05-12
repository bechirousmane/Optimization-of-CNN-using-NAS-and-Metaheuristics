import unittest
import torch
import torch.nn as nn
from .searchSpaceGA import (
    encode_layer, decode_layer, 
    architecture_to_binary, binary_to_architecture,
    is_valid_architecture, SIZE_INDIVIDUAL
)
from ..searchSpaceConfig import Config
from .geneticOperation import (
    generate_valid_architecture,
    build_torch_network,
    elitisteSelection,
    polynomialRankSelection,
    probabilisticTournamentSelection
)

class TestSearchSpaceFunctions(unittest.TestCase):
    
    def test_encode_decode_layer_conv(self):
        """Test encoding and decoding a Conv layer"""
        conv_layer = {
            "type": "Conv",
            "filters": 32,
            "kernel": 3,
            "stride": 1
        }
        encoded = encode_layer(conv_layer)
        self.assertEqual(len(encoded), SIZE_INDIVIDUAL)
        self.assertEqual(encoded[:2], "01")  # Conv type
        decoded = decode_layer(encoded)
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
        encoded = encode_layer(pool_layer)
        self.assertEqual(len(encoded), SIZE_INDIVIDUAL)
        self.assertEqual(encoded[:2], "10")  # Pool type
        decoded = decode_layer(encoded)
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
        encoded = encode_layer(fc_layer)
        self.assertEqual(len(encoded), SIZE_INDIVIDUAL)
        self.assertEqual(encoded[:2], "11")  # FC type
        decoded = decode_layer(encoded)
        self.assertEqual(decoded["type"], fc_layer["type"])
        self.assertEqual(decoded["size"], fc_layer["size"])
        self.assertEqual(decoded["activation"], fc_layer["activation"])
    
    def test_encode_decode_none_layer(self):
        """Test encoding and decoding a None layer"""
        none_layer = None
        encoded = encode_layer(none_layer)
        self.assertEqual(len(encoded), SIZE_INDIVIDUAL)
        self.assertEqual(encoded[:2], "00")  # None type
        decoded = decode_layer(encoded)
        self.assertIsNone(decoded)
    
    def test_architecture_to_binary(self):
        """Test converting architecture to binary string"""
        architecture = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "Pool", "kernel": 5, "stride": 2},
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ]
        binary = architecture_to_binary(architecture)
        self.assertEqual(len(binary), len(architecture) * SIZE_INDIVIDUAL)
        
        # Test individual parts of the binary string
        self.assertEqual(binary[:2], "01")  # Conv
        self.assertEqual(binary[8:10], "10")  # Pool
        self.assertEqual(binary[16:18], "11")  # FC
    
    def test_binary_to_architecture(self):
        """Test converting binary string to architecture"""
        # Create a known binary string
        binary = "01010100" + "10010100" + "11010000"
        
        architecture = binary_to_architecture(binary)
        self.assertEqual(len(architecture), 3)
        
        # Check first layer (Conv)
        self.assertEqual(architecture[0]["type"], "Conv")
        # Check second layer (Pool)
        self.assertEqual(architecture[1]["type"], "Pool")
        # Check third layer (FC)
        self.assertEqual(architecture[2]["type"], "FC")
    
    def test_is_valid_architecture_valid(self):
        """Test valid architectures with the validation function"""
        # Valid minimal architecture: 1 Conv + 1 FC
        valid_arch = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ]
        self.assertTrue(is_valid_architecture(valid_arch))
        
        # Valid complex architecture
        valid_complex = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "Pool", "kernel": 3, "stride": 2},
            {"type": "Conv", "filters": 64, "kernel": 5, "stride": 1},
            {"type": "FC", "size": 128, "activation": nn.ReLU},
            {"type": "FC", "size": 64, "activation": nn.Tanh}
        ]
        self.assertTrue(is_valid_architecture(valid_complex))
    
    def test_is_valid_architecture_invalid(self):
        """Test invalid architectures with the validation function"""
        # Too few layers
        invalid_few = []
        self.assertFalse(is_valid_architecture(invalid_few))
        
        # First layer not Conv
        invalid_first = [
            {"type": "Pool", "kernel": 3, "stride": 2},
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ]
        self.assertFalse(is_valid_architecture(invalid_first))
        
        # No FC layer
        invalid_no_fc = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "Conv", "filters": 64, "kernel": 5, "stride": 1},
            {"type": "Pool", "kernel": 3, "stride": 2}
        ]
        self.assertFalse(is_valid_architecture(invalid_no_fc))
        
        # FC followed by Conv (non-sequential FC)
        invalid_order = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "FC", "size": 64, "activation": nn.ReLU},
            {"type": "Conv", "filters": 64, "kernel": 5, "stride": 1}
        ]
        self.assertFalse(is_valid_architecture(invalid_order))
        
        # Consecutive Pool layers
        invalid_consecutive_pool = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            {"type": "Pool", "kernel": 3, "stride": 2},
            {"type": "Pool", "kernel": 5, "stride": 2},
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ]
        self.assertFalse(is_valid_architecture(invalid_consecutive_pool))
        
        # None layer in architecture
        invalid_none = [
            {"type": "Conv", "filters": 32, "kernel": 3, "stride": 1},
            None,
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ]
        self.assertFalse(is_valid_architecture(invalid_none))
    
    def test_generate_valid_architecture(self):
        """Test the generation of valid architectures"""
        # Generate multiple architectures to ensure they meet all criteria
        for _ in range(10):
            arch = generate_valid_architecture()
            self.assertTrue(is_valid_architecture(arch))
            
            # Check specific rules
            self.assertEqual(arch[0]["type"], "Conv")  # First layer must be Conv
            
            # Find last FC layer index
            last_fc_index = None
            for i, layer in enumerate(arch):
                if layer["type"] == "FC":
                    last_fc_index = i
                    break
            
            # At least one FC layer
            self.assertIsNotNone(last_fc_index)
            
            # All layers after first FC must be FC
            for i in range(last_fc_index, len(arch)):
                self.assertEqual(arch[i]["type"], "FC")
            
            # No consecutive Pool layers
            for i in range(len(arch) - 1):
                if arch[i]["type"] == "Pool":
                    self.assertNotEqual(arch[i+1]["type"], "Pool")
    
    def test_build_torch_network(self):
        """Test building a PyTorch network from an architecture binary string"""
        # Create a simple architecture binary
        arch_binary = architecture_to_binary([
            {"type": "Conv", "filters": 16, "kernel": 3, "stride": 1},
            {"type": "Pool", "kernel": 3, "stride": 2},
            {"type": "FC", "size": 64, "activation": nn.ReLU}
        ])
        
        # Build network with default CIFAR-10 dimensions
        net = build_torch_network(arch_binary, input_shape=(3, 32, 32), num_classes=10)
        
        # Check that the network is a sequential model
        self.assertIsInstance(net, nn.Sequential)
        
        # Create a dummy input and check forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = net(dummy_input)
        
        # Check output dimensions
        self.assertEqual(output.shape, (1, 10))

    def setUp(self):
        """
        Prepare a consistent test dataset for selections tests
        """
        # Example population with different values
        self.population = [
            [1, 2, 3],    # Low fitness
            [4, 5, 6],    # Medium fitness
            [7, 8, 9],    # High fitness
            [10, 11, 12], # Very high fitness
            [13, 14, 15]  # Highest fitness
        ]
        
        # Simple fitness function based on sum
        def simple_fitness(individual, **kwargs):
            return sum(individual)
        
        self.fitness_function = simple_fitness
    
    def test_elitist_selection_basic(self):
        """
        Basic test of elitist selection
        """
        # Select the 2 best architectures
        selected = elitisteSelection(
            self.population, 
            self.fitness_function, 
            nbr=2
        )
        
        # Check that the correct number of individuals is selected
        self.assertEqual(len(selected), 2)
        
        # Check that the best are selected
        expected_best = [self.population[4], self.population[3]]
        self.assertEqual(selected, expected_best)
    
    def test_elitist_selection_edge_cases(self):
        """
        Test edge cases for elitist selection
        """
        # Case where requested number is greater than population
        full_population = elitisteSelection(
            self.population, 
            self.fitness_function, 
            nbr=10
        )
        self.assertEqual(len(full_population), len(self.population))
        
        # Case with an empty population
        empty_population = elitisteSelection(
            [], 
            self.fitness_function, 
            nbr=2
        )
        self.assertEqual(len(empty_population), 0)
    
    # def test_polynomial_rank_selection_distribution(self):
    #     """
    #     Test the distribution of polynomial rank selection
    #     """
    #     # Number of repetitions to verify distribution
    #     num_trials = 10000
    #     selections = []
        
    #     for _ in range(num_trials):
    #         selected = polynomialRankSelection(
    #             self.population, 
    #             self.fitness_function, 
    #             nbr=1,
    #             selective_pressure=1.5
    #         )[0]
    #         selections.append(selected)
        
    #     # Convert selections to indices to analyze distribution
    #     selection_indices = [self.population.index(sel) for sel in selections]
        
    #     # Check that the distribution favors individuals with better fitness
    #     # Last indices (best individuals) should appear more frequently
    #     top_half_count = sum(1 for idx in selection_indices if idx >= len(self.population) // 2)
        
    #     # At least 60% of selections should come from the top half
    #     self.assertGreater(top_half_count / num_trials, 0.6)
    
    def test_polynomial_rank_selection_parameters(self):
        """
        Test parameters of polynomial rank selection
        """
        # Test with different selection pressures
        for pressure in [1.1, 1.5, 2.0]:
            selected = polynomialRankSelection(
                self.population, 
                self.fitness_function, 
                nbr=2,
                selective_pressure=pressure
            )
            
            # Check that the correct number of individuals is selected
            self.assertEqual(len(selected), 2)
    
    def test_probabilistic_tournament_selection_distribution(self):
        """
        Test the distribution of probabilistic tournament selection
        """
        # Number of repetitions to verify distribution
        num_trials = 10000
        selections = []
        
        for _ in range(num_trials):
            selected = probabilisticTournamentSelection(
                self.population, 
                self.fitness_function, 
                nbr=1,
                tournament_size=3,
                tournament_prob=0.75
            )[0]
            selections.append(selected)
        
        # Convert selections to indices to analyze distribution
        selection_indices = [self.population.index(sel) for sel in selections]
        
        # Check that the distribution favors individuals with better fitness
        # Last indices (best individuals) should appear more frequently
        top_half_count = sum(1 for idx in selection_indices if idx >= len(self.population) // 2)
        
        # At least 60% of selections should come from the top half
        self.assertGreater(top_half_count / num_trials, 0.6)
    
    def test_probabilistic_tournament_selection_parameters(self):
        """
        Test parameters of probabilistic tournament selection
        """
        # Test with different tournament sizes
        for tournament_size in [2, 3, 5]:
            selected = probabilisticTournamentSelection(
                self.population, 
                self.fitness_function, 
                nbr=2,
                tournament_size=tournament_size,
                tournament_prob=0.75
            )
            
            # Check that the correct number of individuals is selected
            self.assertEqual(len(selected), 2)
        
        # Test with different tournament probabilities
        for tournament_prob in [0.5, 0.75, 0.9]:
            selected = probabilisticTournamentSelection(
                self.population, 
                self.fitness_function, 
                nbr=2,
                tournament_size=3,
                tournament_prob=tournament_prob
            )
            
            # Check that the correct number of individuals is selected
            self.assertEqual(len(selected), 2)
    
    def test_selection_methods_with_custom_fitness(self):
        """
        Test selection methods with a custom fitness function
        """
        # More complex fitness function
        def complex_fitness(individual, weight=1.0, **kwargs):
            return sum(individual) * weight
        
        # Test for each selection method
        selection_methods = [
            elitisteSelection,
            polynomialRankSelection,
            probabilisticTournamentSelection
        ]
        
        for method in selection_methods:
            selected = method(
                self.population, 
                complex_fitness,
                nbr=2,
                weight=1.5  # Custom parameter
            )
            
            # Check that the correct number of individuals is selected
            self.assertEqual(len(selected), 2)
    


if __name__ == '__main__':
    unittest.main()