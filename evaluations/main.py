"""
Main execution script for CIFAR10 architecture search
"""

import torch
import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from search_spaces.searchSpaceConfig import Config
from architecture_search import ArchitectureSearch
from utils.logger import setup_logger


def save_results(search_type, best_arch, best_fitness, history, accuracy, loss, path="./results"):
    """Save search results to disk.
    
    Args:
        search_type (str): Type of search algorithm used
        best_arch (dict): Best architecture found
        best_fitness (float): Fitness score of the best architecture
        history (list): History of fitness scores during search
        accuracy (float): Final test accuracy
        loss (float): Final test loss
        path (str): Path to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{path}/{search_type}_search_{timestamp}.json"
    
    # Prepare results dictionary
    results = {
        "search_type": search_type,
        "best_architecture": best_arch,
        "best_fitness": best_fitness,
        "history": history,
        "final_accuracy": float(accuracy),
        "final_loss": float(loss),
        "timestamp": timestamp
    }
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filename}")
    return filename


def plot_comparison(results, title="Comparison of Search Algorithms", save_path="./results"):
    """Plot comparison of different search algorithms.
    
    Args:
        results (dict): Dictionary of results for each algorithm
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for algo, history in results.items():
        plt.plot(history, label=f"{algo} search")
    
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{save_path}/algorithm_comparison_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Comparison plot saved to {filename}")


def main():
    """Main program function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Architecture Search for CIFAR10')
    parser.add_argument('algorithm', type=str, choices=['random', 'genetic', 'firefly', 'all'],
                        help='Search algorithm to execute (random, genetic, firefly, all)')
    parser.add_argument('--config', type=str, default="../config.json",
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for search phase')
    parser.add_argument('--final-epochs', type=int, default=30,
                        help='Number of epochs for final training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--population', type=int, default=20,
                        help='Population size')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Number of search iterations')
    parser.add_argument('--sub-train', type=int, default=20000,
                        help='Number of training samples for optimization')
    parser.add_argument('--sub-test', type=int, default=5000,
                        help='Number of test samples for optimization')
    parser.add_argument('--output-dir', type=str, default="./results",
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Add algorithm-specific parameters
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation rate for genetic algorithm')
    parser.add_argument('--tournament-size', type=int, default=5,
                        help='Tournament size for genetic algorithm')
    parser.add_argument('--crossover-prob', type=float, default=.75,
                       help='Crossover probability for genetic algorithm')
    
    
    # Add firefly-specific parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha parameter for firefly algorithm')
    parser.add_argument('--beta0', type=float, default=1.0,
                        help='Beta0 parameter for firefly algorithm')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma parameter for firefly algorithm')
    parser.add_argument('--data',type=str,default='mnist',
                        help='The data for training')
    parser.add_argument('--input-dim', type=tuple, default=(1,28,28),
                        help='The input shape')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(f"{args.algorithm}_search_{args.data}", args.output_dir)
    logger.info("Starting architecture search")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Custom search parameters based on command line arguments
    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'population_size': args.population,
        'iterations': args.iterations,
        'data' : args.data,
        'input_shape': args.input_dim,
        'optimizer': "AdamW",
        'num_classes': 10,
        'seed': args.seed
    }
    
    # Initialize architecture search
    arch_search = ArchitectureSearch(
        config_path=args.config,
        params=params
    )
    
    # Display current configuration
    logger.info("Loaded configuration:")
    logger.info(Config.get_config_as_dict())
    
    # Load data
    logger.info(f"Loading {args.data} data...")
    arch_search.load_data(data_name=params['data'],n_sub_train=args.sub_train, n_sub_test=args.sub_test)
    
    # Create directories for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store results for comparison
    all_results = {}
    
    # Run all selected algorithms
    algorithms_to_run = ['random', 'genetic', 'firefly'] if args.algorithm == 'all' else [args.algorithm]
    
    for algo in algorithms_to_run:
        # Execute specified algorithm
        if algo == 'random':
            logger.info("\n=== Random Search ===")
            best_arch, best_fitness, history = arch_search.run_random_search()
            search_type = 'random'
            
        elif algo == 'genetic':
            logger.info("\n=== Genetic Search ===")
            best_arch, best_fitness, history = arch_search.run_genetic_search(
                selection_type="elitist",
                mutation_rate=args.mutation_rate,
                tournament_size=args.tournament_size,
                tournament_prob=0.75,
                selection_presure=1.5,
                crossover_prob=args.crossover_prob
            )
            search_type = 'genetic'
            
        elif algo == 'firefly':
            logger.info("\n=== Firefly Search ===")
            best_arch, best_fitness, history = arch_search.run_firefly_search(
                alpha=args.alpha,
                beta0=args.beta0,
                gamma=args.gamma
            )
            search_type = 'firefly'
        
        # Store history for comparison
        all_results[search_type] = history
        
        # Display best architecture
        logger.info(f"Best architecture found ({search_type} search):")
        logger.info(best_arch)
        
        # Train the best model
        logger.info(f"\n=== Training best model ({search_type} search) ===")
        accuracy, loss = arch_search.train_best_model(
            search_type=search_type, 
            epochs=args.final_epochs
        )
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test loss: {loss:.4f}")
        
        # Save results
        save_results(
            search_type=search_type,
            best_arch=best_arch,
            best_fitness=best_fitness,
            history=history,
            accuracy=accuracy,
            loss=loss,
            path=args.output_dir
        )
        
        # Display training curve
        arch_search.plot_training_history(
            search_type=search_type,
            title=f"Courbe de convergence trouver par l'algorithme {search_type} sur {params['data']}",
            save_path=os.path.join(args.output_dir, f"{search_type}_training_{params['data']}_history.png")
        )
        
        # Display confusion matrix
        arch_search.plot_confusion_matrix(
            search_type=search_type,
            save_path=os.path.join(args.output_dir, f"{search_type}_confusion_matrix_{params['data']}.png")
        )
    
    # Plot comparison if multiple algorithms were run
    if len(all_results) > 1:
        plot_comparison(all_results, title="Comparison of Search Algorithms", save_path=args.output_dir)


if __name__ == "__main__":
    # Check GPU availability
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"GPU used: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # Run main program
        main()
    except Exception as e:
        # Setup basic logger for uncaught exceptions
        import logging
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger("error_logger")
        logger.error(f"Uncaught exception: {str(e)}", exc_info=True)
        raise