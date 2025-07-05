import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data.cifar10_loader import load_cifar10
from data.mnist_loader import load_mnist
from optimizers.random_search import RandomSearch
from optimizers.genetic_search import GeneticSearch
from optimizers.firefly_search import FireFlySearch
from search_spaces.searchSpaceConfig import Config
from train.trainer import ModelTrainer
from search_spaces.utils import build_torch_network


class ArchitectureSearch:
    """Main class for neural network architecture research."""
    
    def __init__(self, config_path=None, params=None):
        """
        Initialization with configuration and parameters.

        Args:
            config_path (str, optional): Path to the configuration file.
            params (dict, optional): Search parameters.
        """
        # Loading the configuration
        if config_path:
            Config.load_config(config_path=config_path)
        
        # Default settings
        self.default_params = {
            'epochs': 10,
            'batch_size': 128,
            'lr': 0.001,
            'population_size': 20,
            'iterations': 15,
            'input_shape': (3, 32, 32),
            'optimizer': "AdamW",
            'num_classes': 10,
            'use_gpu': torch.cuda.is_available(),
            'device': "cuda:0" if torch.cuda.is_available() else "cpu"
        }
        
        # Merge with the provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
    
        self.sub_train = None
        self.sub_test = None
        self.class_names = None
        
        self.best_architectures = {}
        self.best_fitnesses = {}
        self.search_histories = {}
        self.avg_fitness_history = {}
        self.trained_models = {}
        self.training_histories = {}
        self.count_eval = {}
        self.invalid_arch_history = {}
    
    def load_data(self, data_name):
        """
            Loads the subset for optimization.

            Args:
                data_name (str) : Data name. must be cifar for CIFAR-10 data or mnist for MNIST data
        """
        print(self.params)
        
        if data_name == 'mnist' : 
            self.sub_train, self.sub_test, self.class_names = load_mnist(
            n_train=self.params["sub_train"], 
            n_test=self.params["sub_test"], 
            n_class=self.params['num_classes'],
            batch_size=self.params['batch_size']
            )
        
        elif data_name == 'cifar' :
            self.sub_train, self.sub_test, self.class_names = load_cifar10(
                n_train=self.params["sub_train"], 
                n_test=self.params["sub_test"], 
                n_class=self.params['num_classes'],
                batch_size=self.params['batch_size']
            )
            
        
        else :
            raise ValueError("data name unsupported.")
    
    def run_random_search(self):
        """
        Performs a random architecture search.

        Returns:
            tuple: Optimal architecture, fitness, history
        """
        random_search = RandomSearch(
            population_size=self.params['population_size'],
            iterations=self.params['iterations'],
            train_loader=self.sub_train,
            test_loader=self.sub_test,
            input_shape=self.params['input_shape'],
            num_classes=self.params['num_classes'],
            epochs=self.params['epochs'],
            lr=self.params['lr'],
            optimizer=self.params['optimizer'],
            use_gpu=self.params['use_gpu']
        )
        
        best_arch, best_fitness, history = random_search.run_search()
        
        self.best_architectures['random'] = best_arch
        self.best_fitnesses['random'] = best_fitness
        self.search_histories['random'] = history
        self.count_eval['random'] = random_search.count_eval
        self.avg_fitness_history['random'] = random_search.avg_fitness_history
        self.invalid_arch_history['random'] = random_search.invalid_arch_history
        
        return best_arch, best_fitness, history
    
    def run_genetic_search(self, selection_type="tournement", mutation_rate=0.1,
                          tournament_size=5, tournament_prob=0.75, selection_presure=1.5, crossover_prob=.75):
        """
        Performs a genetic architecture search.

        Args:
            selection_type (str): Selection type.
            mutation_rate (float): Mutation rate.
            tournament_size (int): Tournament size.
            tournament_prob (float): Tournament probability.
            selection_presure (float): Selection pressure.
            
        Returns:
            tuple: Optimal architecture, fitness, history
        """
        genetic_search = GeneticSearch(
            selection_type=selection_type,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            tournament_prob=tournament_prob,
            crossover_prob=crossover_prob,
            selection_presure=selection_presure,
            population_size=self.params['population_size'],
            iterations=self.params['iterations'],
            train_loader=self.sub_train,
            test_loader=self.sub_test,
            input_shape=self.params['input_shape'],
            num_classes=self.params['num_classes'],
            epochs=self.params['epochs'],
            lr=self.params['lr'],
            optimizer=self.params['optimizer'],
            use_gpu=self.params['use_gpu']
        )
        
        best_arch, best_fitness, history = genetic_search.run_search()
        
        self.best_architectures['genetic'] = best_arch
        self.best_fitnesses['genetic'] = best_fitness
        self.search_histories['genetic'] = history
        self.count_eval['genetic'] = genetic_search.count_eval
        self.avg_fitness_history['genetic'] = genetic_search.avg_fitness_history
        self.invalid_arch_history['genetic'] = genetic_search.invalid_arch_history
        
        return best_arch, best_fitness, history
    
    def run_firefly_search(self, alpha, beta0, gamma, sigma0, prob):
        """
        Performs a search using the firefly algorithm.

        Args:
            alpha : float, control parameter
            beta0 : float, the attractiveness at distance zero,
            gamma : float, the light absorption coefficient
            sigma0 : float, Variance
            logger : logging.Logger
            prob : float, the probability of using a normal distribution for firefly movement
            
        Returns:
            tuple: Optimal architecture, fitness, history
        """
        firefly_search = FireFlySearch(
            population_size=self.params['population_size'],
            iterations=self.params['iterations'],
            train_loader=self.sub_train,
            test_loader=self.sub_test,
            input_shape=self.params['input_shape'],
            num_classes=self.params['num_classes'],
            epochs=self.params['epochs'],
            lr=self.params['lr'],
            optimizer=self.params['optimizer'],
            use_gpu=self.params['use_gpu'],
            alpha=alpha,
            beta0=beta0,
            gamma=gamma,
            sigma0=sigma0,
            prob=prob,
        )
        
        best_arch, best_fitness, history = firefly_search.run_search()
        
        self.best_architectures['firefly'] = best_arch
        self.best_fitnesses['firefly'] = best_fitness
        self.search_histories['firefly'] = history
        self.count_eval['firefly'] = firefly_search.count_eval
        self.avg_fitness_history['firefly'] = firefly_search.avg_fitness_history
        self.invalid_arch_history['firefly'] = firefly_search.invalid_arch_history
        
        return best_arch, best_fitness, history
    
    def train_best_model(self, search_type='random', epochs=30, verbose=True):
        """
        Trains the best model found on the full dataset.

        Args:
            search_type (str): Search type ('random', 'genetic', 'firefly').
            epochs (int): Number of training epochs.
            verbose (bool): Display training details.
            
        Returns:
            tuple: Test accuracy, test loss
        """
        if search_type not in self.best_architectures:
            raise ValueError(f"No architecture found for search type '{search_type}'")
        
        best_arch = self.best_architectures[search_type]
        
        best_model = build_torch_network(
            best_arch, 
            input_shape=self.params['input_shape'], 
            num_classes=self.params['num_classes']
        )
        
        model_trainer = ModelTrainer(
            best_model,
            device=self.params['device'],
            lr=self.params['lr'],
            epochs=epochs,
            train_loader=self.sub_train,
            test_loader=self.sub_test,
            optimizer=self.params['optimizer']
        )
        
        train_loss, train_accuracy = model_trainer.train(verbose=verbose)
        test_accuracy, test_loss = model_trainer.test()
        
        self.trained_models[search_type] = best_model
        self.training_histories[search_type] = model_trainer.loss_history
        
        return train_accuracy, train_loss, test_accuracy, test_loss
    
    def plot_training_history(self, search_type='random', title=None, save_path=None):
        """
        Displays the model's training history.

        Args:
            search_type (str): Search type.
            title (str, optional): Chart title.
            save_path (str, optional): Path to save the chart.
        """
        if search_type not in self.training_histories:
            raise ValueError(f"No training history for the type'{search_type}'")
        
        loss_history = self.training_histories[search_type]
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(len(loss_history)), loss_history)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Courbe de convergence du meilleur modèle trouvé par la recherche {search_type}")
            
        plt.xlabel("Époque")
        plt.ylabel("Perte")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else : 
            plt.show()
    
    def plot_confusion_matrix(self, search_type='random', save_path=None):
        """
        Displays the model's confusion matrix on the test data.

        Args:
            search_type (str): Search type.
            save_path (str, optional): Path to save the graph.
        """
        if search_type not in self.trained_models:
            raise ValueError(f"No model trained for type '{search_type}'")
        
        model = self.trained_models[search_type]
        device = self.params['device']
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.sub_test:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.class_names)
        
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matrice de confusion pour le modèle {search_type}")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else :
            plt.show()

    def plot_invalid_arch_hist(self, search_type="random", title = None, save_path=None) :
        """
            Displays the model's training history.

            Args:
                search_type (str): Search type.
                title (str, optional): Chart title.
                save_path (str, optional): Path to save the chart.
        """
        if search_type not in self.invalid_arch_history:
            raise ValueError(f"No invalid architecture history for the type'{search_type}'")
        
        invalid_arch_history = self.invalid_arch_history[search_type]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(invalid_arch_history)), invalid_arch_history)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Courbe d'evolution du nombre d'architecture invalide pour l'algorithme {search_type}")
            
        plt.xlabel("generation")
        plt.ylabel("invalid arch nbr")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()

        else :
            plt.show()

    
    def compare_search_methods(self, methods=None):
        """
        Compares different search methods.

        Args:
            methods (list, optional): List of methods to compare.
        """
        if not methods:
            methods = list(self.best_fitnesses.keys())
        
        for method in methods:
            if method not in self.best_fitnesses:
                raise ValueError(f"Méthode '{method}' non trouvée dans les résultats")
        
        print("Comparaison des méthodes de recherche d'architecture:")
        print("-" * 50)
        print(f"{'Méthode':<15} | {'Meilleure fitness':<20} | {'Précision de test':<15}")
        print("-" * 50)
        
        for method in methods:
            fitness = self.best_fitnesses[method]
            _,_,accuracy,_ = self.train_best_model(search_type=method, verbose=False)
            print(f"{method:<15} | {fitness:<20.4f} | {accuracy:<15.4f}")

