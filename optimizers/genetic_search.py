import asyncio
import torch
import numpy as np
from search_spaces.utils import generate_valid_architecture, is_valid_architecture, build_torch_network
from search_spaces.genetic.geneticOperation import *
from search_spaces.genetic.searchSpaceGA import *
from train.trainer import ModelTrainer
from ressource.ressource_manager import ResourceManager

class GeneticSearch:
    def __init__(self, 
                 selection_type = "elitist",
                 mutation_rate = 0.1,
                 tournament_size = 5,
                 tournament_prob=.75,
                 selection_presure=1.5,
                 crossover_prob = .75,
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
            selection_type : str, name of the selection type : eletist, polynomial, tournement
            mutation_rate : float, the mutation rate 
            tournament_size : float, the tournement size when tournement selection is used
            tournament_prob :float, the tournement probability when tournement selection is used
            selection_presure : float, the selection presure when polynomiale rank selection is used.
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
        self.seletion_type = selection_type
        self.selection_presure = selection_presure
        self.tournement_size = tournament_size
        self.tournement_prob = tournament_prob
        self.mutation_rate = mutation_rate
        self.crossover_prob = crossover_prob
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
            self.count_eval += 1
            
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
                #print(device)
                fitness = await self.evaluate_architecture(architecture, device)
                fitness_scores[index] = (architecture,fitness)
                return index, fitness, architecture
        
        # Create tasks for each architecture in the population
        tasks = [evaluate_with_resource(i, arch) for i, arch in enumerate(population)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Update history and check for new best architecture
        for index, fitness, architecture in results:
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_architecture = architecture
                print(f"New best architecture found with fitness: {fitness}")
           
            self.history.append(self.best_fitness)

        
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
    
    def _selection(self, population, fitness_scores, n) :
        if self.seletion_type == "elitist" :
            return elitisteSelection(population, fitness_scores,n)
        elif self.seletion_type == "polynomial" : 
            return polynomialRankSelection(population, fitness_scores,n,self.selection_presure)
        elif self.seletion_type == "tournement" :
            return probabilisticTournamentSelection(population,fitness_scores, n, self.tournement_size, self.tournement_prob)
        else :
            raise ValueError("type of selection Unsupported")

    async def search(self) :
        """
        Perform the genetic algorithm for optimal CNN architectures.
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        self.history = []
        # Generate initial population
        current_population = self.generate_population()

        # Evaluate initial population
        population_with_fitness = await self.evaluate_population(current_population)
            
        for iteration in range(self.iterations):
            print(f"Iteration {iteration+1}/{self.iterations}")
            
            # Create new generation
            new_generation = []
            
            # Elitism: Keep the best individuals
            elite_count = max(1, self.population_size // 10)  # 10% elitism
            sorted_population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            elite = [arch for arch, _ in sorted_population[:elite_count]]
            new_generation.extend(elite)
            
            # Fill the rest with crossover and mutation
            while len(new_generation) < self.population_size:
                # Select parents
                parents = self._selection(current_population, population_with_fitness, 2)
                
                # Apply crossover with probability 0.7
                if np.random.random() < self.crossover_prob:
                    child_binary = onePointCrossover(
                        architecture_to_binary(parents[0]), 
                        architecture_to_binary(parents[1])
                    )
                    child = binary_to_architecture(child_binary)
                else:
                    # If no crossover, just take one parent
                    child = parents[0]
                
                child_binary = architecture_to_binary(child)
                mutated_binary = mutate(child_binary, self.mutation_rate)
                mutated_child = binary_to_architecture(mutated_binary)
                
                if is_valid_architecture(mutated_child):
                    new_generation.append(mutated_child)
            
            # Trim to population size if needed
            new_generation = new_generation[:self.population_size]
            
            # Evaluate new generation
            population_with_fitness = await self.evaluate_population(new_generation)
            
            # Print statistics
            current_fitness = [f for _, f in population_with_fitness]
            avg_fitness = sum(current_fitness) / len(current_fitness)
            print(f"Average fitness: {avg_fitness}")
            print(f"Best fitness so far: {self.best_fitness}")
            print(f"Models evaluated: {self.count_eval}")
        
        return self.best_architecture, self.best_fitness, self.history


    def run_search(self):
        """
        Run the search process (synchronous wrapper for async search).
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.search())