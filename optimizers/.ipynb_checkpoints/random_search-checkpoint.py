import asyncio
import torch
import numpy as np
from search_spaces.utils import generate_valid_architecture, is_valid_architecture, build_torch_network
from train.trainer import ModelTrainer
from ressource.ressource_manager import ResourceManager

class RandomSearch:
    def __init__(self, 
                 population_size=50, 
                 iterations=5, 
                 train_loader=None, 
                 test_loader=None, 
                 input_shape=(3, 32, 32),
                 num_classes=10,
                 epochs=5,
                 lr=0.001,
                 optimizer="Adam",
                 use_gpu=True,
                 max_concurrent=None):
        """
        Initialize the random search for CNN architectures.
        
        Args:
            population_size: int, size of the population for each iteration
            iterations: int, number of search iterations
            train_loader: DataLoader, training data loader
            test_loader: DataLoader, testing data loader
            input_shape: tuple, input shape (channels, height, width)
            num_classes: int, number of output classes
            epochs: int, number of training epochs for each model
            lr: float, learning rate for training
            optimizer: str, optimizer to use (Adam or AdamW)
            use_gpu: bool, whether to use GPU if available
            max_concurrent: int, maximum number of concurrent evaluations
        """
        self.population_size = population_size
        self.iterations = iterations
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.resource_manager = ResourceManager(use_gpu=use_gpu, max_concurrent=max_concurrent)
        
        # Store the best architecture and its fitness
        self.best_architecture = None
        self.best_fitness = float('-inf')  # Higher fitness is better
        
        # History of all evaluated architectures and their fitness
        self.history = []

    async def evaluate_architecture(self, architecture, device):
        """
        Evaluate a single architecture.
        
        Args:
            architecture: list of dict, the architecture to evaluate
            device: str, the device to use for evaluation
            
        Returns:
            float: fitness score normalized between 0 and 1
                  or 0 if the architecture is invalid
        """
        if not is_valid_architecture(architecture):
            return 0.0
        
        try:
            # Build the network
            model = build_torch_network(
                architecture, 
                input_shape=self.input_shape, 
                num_classes=self.num_classes
            )
            
            # Train and evaluate the model
            trainer = ModelTrainer(
                model=model,
                device=device,
                lr=self.lr,
                epochs=self.epochs,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                optimizer=self.optimizer
            )
            
            # Train the model
            trainer.train(verbose=False)
            
            # Get the last training loss
            train_loss = trainer.loss_history[-1] if trainer.loss_history else 0
            
            # Test the model
            accuracy, test_loss = trainer.test()
            
            avg_loss = (train_loss + test_loss) / 2
            fitness = 1/(1 + avg_loss)  
            
            return fitness
            
        except Exception as e:
 #           print(f"Error evaluating architecture: {e}")
            return 0.0

    async def evaluate_population(self, population):
        """
        Evaluate a population of architectures in parallel.
        
        Args:
            population: list of list of dict, architectures to evaluate
            
        Returns:
            list of float: fitness scores for each architecture
        """
        fitness_scores = [0.0] * len(population)
        
        async def evaluate_with_resource(index, architecture):
            async with self.resource_manager.acquire() as device:
                fitness = await self.evaluate_architecture(architecture, device)
                fitness_scores[index] = fitness
                return index, fitness, architecture
        
        # Create tasks for each architecture in the population
        tasks = [evaluate_with_resource(i, arch) for i, arch in enumerate(population)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Update history and check for new best architecture
        for index, fitness, architecture in results:
            self.history.append((architecture, fitness))
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_architecture = architecture
                print(f"New best architecture found with fitness: {fitness}")
        
        return fitness_scores

    def generate_population(self):
        """
        Generate a population of random valid architectures.
        
        Returns:
            list of list of dict: a list of generated architectures
        """
        population = []
        for _ in range(self.population_size):
            arch = generate_valid_architecture()
            population.append(arch)
        return population

    async def search(self):
        """
        Perform the random search for optimal CNN architectures.
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        for iteration in range(self.iterations):
            print(f"Iteration {iteration+1}/{self.iterations}")
            
            # Generate population
            population = self.generate_population()
            
            # Evaluate population
            fitness_scores = await self.evaluate_population(population)
            
            # Print statistics
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Average fitness: {avg_fitness}")
            print(f"Best fitness so far: {self.best_fitness}")
        
        return self.best_architecture, self.best_fitness, self.history

    def run_search(self):
        """
        Run the search process (synchronous wrapper for async search).
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.search())