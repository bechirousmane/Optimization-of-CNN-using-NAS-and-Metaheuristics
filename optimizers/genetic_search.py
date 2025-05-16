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
                 tournement_size = 5,
                 tournement_prob=.75,
                 selection_presure=1.5,
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
            tournement_size : float, the tournement size when tournement selection is used
            tournement_prob :float, the tournement probability when tournement selection is used
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
        self.tournement_size = tournement_size
        self.tournement_prob = tournement_prob
        self.mutation_rate = mutation_rate
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

    async def search(self):
        """
        Perform the genetic algorithm for optimal CNN architectures.
        
        Returns:
            tuple: (best_architecture, best_fitness, history)
        """
        self.history = []
        # Generate population
        population = self.generate_population()

        # Evaluate population
        fitness_scores = await self.evaluate_population(population)
            
        for iteration in range(self.iterations):
            print(f"Iteration {iteration+1}/{self.iterations}")

            new_generation = []

            #Selection parent
            parent1, parent2 = self._selection(population, fitness_scores, 2)

            # Crossover
            child = onePointCrossover(architecture_to_binary(parent1),architecture_to_binary(parent2))
            new_generation.append(binary_to_architecture(child))

            # Mutation 
            for idx, individual in enumerate(population) :
                mutant = mutate(architecture_to_binary(individual),self.mutation_rate)
                if mutant != individual :
                    new_generation.append(binary_to_architecture(mutant))
                    population.append(binary_to_architecture(mutant))
            
            # Evaluate new generation
            new_fitness_scores = await self.evaluate_population(new_generation)

            fitness_scores.extend(new_fitness_scores)

            
            # Print statistics
            avg_fitness = sum(score for _,score in fitness_scores) / len(fitness_scores)
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