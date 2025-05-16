import asyncio
import torch
import numpy as np
from search_spaces.utils import generate_valid_architecture, is_valid_architecture, build_torch_network
from search_spaces.firefly.searchSpaceFA import *
from search_spaces.firefly.fireflyOperation import * 
from train.trainer import ModelTrainer
from ressource.ressource_manager import ResourceManager

class FireFlySearch:
    def __init__(self, 
                 alpha,
                 beta0,
                 gamma,
                 sigma0,
                 prob,
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
        Initialize the genetic algorithme search for CNN architectures.
        
        Args:
            alpha : float, control parameter
            beta0 : float, the attractiveness at distance zero,
            gamma : float, the light absorption coefficient
            sigma0 : float, Variance
            prob : float, the probability of using a normal distribution for firefly movement
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
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.sigma0 = sigma0
        self.prob = prob
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
        self.history = []
        self.best_fitness = 0
        self.count_eval = 0
        self.best_architecture = None

    
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
            self.count_eval += 1 
            
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
                fitness_scores[index] = (architecture,fitness)
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

    async def search(self):
        """
        Perform the firefly algorithm for optimal CNN architectures.
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        self.history = []
        # Generate population
        population = initializeFireflyPopulation(self.population_size)

        # Evaluate initial population
        fitness_results = await self.evaluate_population(population)
        
        # Convert to format [(architecture, fitness), ...]
        fireflies = [(arch, fitness) for arch, fitness in fitness_results]
        
        for iteration in range(self.iterations):
            print(f"Iteration {iteration+1}/{self.iterations}")
            
            # Adjust alpha for gradual reduction (exploration to exploitation)
            current_alpha = self.alpha * (0.97 ** iteration)
            
            # Reduce sigma over iterations for better convergence
            current_sigma = self.sigma0 * (1 - iteration/self.iterations)
            
            # Sort fireflies by brightness (fitness)
            fireflies.sort(key=lambda x: x[1], reverse=True)
            
            new_generation = []
            
            for i in range(len(fireflies) - 1):
                arch_i, fitness_i = fireflies[i]
                firefly_i = architecture_to_vector(arch_i)
                
                # Compare with all brighter fireflies
                for j in range(i):
                    arch_j, fitness_j = fireflies[j]
                    
                    # Only move towards brighter fireflies
                    if fitness_j > fitness_i:
                        brightest_firefly = architecture_to_vector(arch_j)
                        
                        # Move firefly i towards firefly j using standard movement
                        new_firefly = moveFirefly(
                            alpha=current_alpha,
                            beta0=self.beta0,
                            gamma=self.gamma,
                            firefly=firefly_i, 
                            brightest_firefly=brightest_firefly
                        )
                        
                        # With probability prob, use multivariate normal distribution for diversity
                        if random.random() <= self.prob:
                            # Create covariance matrix - diagonal matrix with current_sigma values
                            cov_matrix = np.eye(len(new_firefly)) * current_sigma
                            
                            # Generate random perturbation using multivariate normal distribution
                            perturbation = np.random.multivariate_normal(
                                mean=brightest_firefly,  
                                cov=cov_matrix                   
                            )
                            
                            # Apply perturbation to the firefly
                            new_firefly = new_firefly + perturbation
                            
                            # Ensure integer values
                            new_firefly = np.round(new_firefly).astype(int)
                            
                            # Validate the firefly to ensure it represents a valid architecture
                            new_firefly = validateFirefly(new_firefly)
                        
                        if not np.array_equal(new_firefly, firefly_i):
                            new_arch = vector_to_achitecture(new_firefly)
                            new_generation.append(new_arch)
            
            # Evaluate new fireflies
            print(f"Size of new generation : {len(new_generation)}")
            if new_generation:
                new_fitness_results = await self.evaluate_population(new_generation)
                new_fireflies = [(arch, fitness) for arch, fitness in new_fitness_results]
                
                # Combine populations and keep only the best
                all_fireflies = fireflies + new_fireflies
                all_fireflies.sort(key=lambda x: x[1], reverse=True)
                fireflies = all_fireflies[:self.population_size]  # Keep population size constant
                
                # Optional: Add random new fireflies if diversity is too low
                if iteration < self.iterations - 1:  # Skip in the last iteration for convergence
                    unique_fitnesses = len(set(fitness for _, fitness in fireflies[:10]))
                    if unique_fitnesses < 3:  # If top 10 have less than 3 different fitness values
                        print("Low diversity detected, adding random fireflies")
                        random_population = initializeFireflyPopulation(max(5, self.population_size//10))
                        random_fitness_results = await self.evaluate_population(random_population)
                        random_fireflies = [(arch, fitness) for arch, fitness in random_fitness_results]
                        
                        # Replace worst performing fireflies with random ones
                        if random_fireflies:
                            fireflies = fireflies[:self.population_size-len(random_fireflies)] + random_fireflies
                            fireflies.sort(key=lambda x: x[1], reverse=True)
            
            # Print statistics
            avg_fitness = sum(fitness for _, fitness in fireflies) / len(fireflies)
            best_current = max(fireflies, key=lambda x: x[1])
            print(f"Average fitness: {avg_fitness}")
            print(f"Best fitness in this iteration: {best_current[1]}")
            print(f"Best fitness overall: {self.best_fitness}")
        
        return self.best_architecture, self.best_fitness, self.history

    def run_search(self):
        """
        Run the search process (synchronous wrapper for async search).
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.search())