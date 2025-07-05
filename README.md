# Optimization-of-CNN-using-NAS-and-Metaheuristics

## Summary :

This project provides a **Python implementation of a pipeline for searching convolutional neural network (CNN) architectures** using **bio-inspired optimization algorithms**. The goal is to explore different optimization methods to automatically adjust certain **structural hyperparameters** of a CNN, within the framework of *Neural Architecture Search* (NAS).

Three types of search algorithms are used:

- **Random Search(Which serves as a basis for comparison)** 

- **Genetic Algorithm (GA)**

- **Firefly Algorithm**, inspired by the behavior of fireflies

The defined search space focuses on several structural aspects of the network:

- The **number of filters** in convolutional layers

- The **size of filters**

- The **presence of pooling layers**

- The **strides** in convolutional and pooling layers

- The **activation function** and **number of neurons** in fully connected layers

The project sets up a complete pipeline, from defining the search space to running the optimization algorithms. It is mainly intended to test and compare different search strategies over a well-defined space, in a reproducible and extensible setup.

The search space is deliberately limited in size to maintain a reasonable computational cost. Although fully configurable, the current implementation defines a **fixed but customizable set of candidate values** for each hyperparameter. For example:

- **Filter sizes**: 4 possible values

- **Number of filters**: 4 possible values

- **Stride values**: 4 possible values

- **Activation functions**: 4 different types

- **Number of neurons in fully connected layers**: 16 possible values

Some architectural parameters are **not included in the optimization process** and are set to fixed values, such as:

- **Padding** in convolutional and pooling layers

- **Minimum and maximum number of layers**

These choices make the search space manageable while allowing for flexible experimentation through configuration changes.

---

## Requirement

Before running this project, make sure you have the following Python packages installed:

- [`PyTorch`](https://pytorch.org/) – for building and training the neural networks

- [`NumPy`](https://numpy.org/) – for numerical operations and array manipulation

- [`Matplotlib`](https://matplotlib.org/) – for plotting and visualizing results

- [`Scikit-learn`](https://scikit-learn.org/) – for additional metrics and data preprocessing

---

## Installation

Clone this repository to your local machine:

```bash
git clone git@github.com:bechirousmane/Optimization-of-CNN-using-NAS-and-Metaheuristics.git
cd Optimization-of-CNN-using-NAS-and-Metaheuristics
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Project Structure

This project follows a modular and organized structure, supporting parallel execution and customizable search configurations:

```bash
.
├── data/                          # Dataset loading modules
│   ├── cifar10_loader.py         # CIFAR-10 dataset loader
│   ├── mnist_loader.py           # MNIST dataset loader
│   └── __init__.py
│
├── evaluations/                  # NAS experiment logic
│   ├── architecture_search.py    # Core logic for architecture evaluation
│   ├── main.py                   # Main entry point to launch experiments
│   └── utils/
│       ├── logger.py             # Logging utility
│       └── __init__.py
│
├── optimizers/                   # Search algorithm implementations
│   ├── firefly_search.py         # Firefly Algorithm
│   ├── genetic_search.py         # Genetic Algorithm
│   ├── random_search.py          # Random Search
│   └── __init__.py
│
├── ressource/                    # Resource manager for parallel execution
│   ├── ressource_manager.py      # Coroutine-based queue for scheduling
│   │                              # model evaluations across CPU/GPU resources
│   └── __init__.py
│
├── search_spaces/                # Search space definitions and configurations
│   ├── firefly/
│   │   ├── fireflyOperation.py   # Firefly-specific encoding and operations
│   │   ├── fireflyTest.py        # Unit test for Firefly search
│   │   ├── searchSpaceFA.py      # Firefly-specific search space config
│   │   └── __init__.py
│   ├── genetic/
│   │   ├── geneticOperation.py   # Genetic-specific encoding and operations
│   │   ├── geneticTest.py        # Unit test for Genetic search
│   │   ├── searchSpaceGA.py      # Genetic-specific search space config
│   │   └── __init__.py
│   ├── searchSpaceConfig.py      # Global config interface for all search spaces
│   ├── utils.py                 
│   └── __init__.py
│
├── train/                        # Model training modules
│   ├── trainer.py                # Defines training loop for candidate models
│   └── __init__.py
│
├── config.json                   # User-defined search configuration (fixed & variable params)
├── requirements.txt              # Project dependencies
```

---

## How to Run

You can launch a Neural Architecture Search (NAS) experiment using the `main.py` script. The script supports multiple search algorithms and offers extensive configurability through command-line arguments or a JSON configuration file.

```bash
python3 evaluations/main.py [ALGORITHM] [OPTIONS]
```

- **`ALGORITHM`**: Specifies the search algorithm to use. Choices are:
  
  - `random`
  
  - `genetic`
  
  - `firefly`
  
  - `all` (runs all algorithms sequentially)

### Example

```bash
python3 evaluations/main.py genetic --config config.json --epochs 10 --final-epochs 30 --population 20 --iterations 15 --batch-size 128 --data mnist
```

This command initiates a genetic algorithm-based NAS on the MNIST dataset, using the specified parameters.

### Command-Line Arguments

| Argument         | Type  | Default          | Description                                                         |
| ---------------- | ----- | ---------------- | ------------------------------------------------------------------- |
| `ALGORITHM`      | str   | —                | Search algorithm to execute (`random`, `genetic`, `firefly`, `all`) |
| `--config`       | str   | `../config.json` | Path to the configuration file                                      |
| `--epochs`       | int   | `10`             | Number of epochs for the search phase                               |
| `--final-epochs` | int   | `30`             | Number of epochs for the final training phase                       |
| `--batch-size`   | int   | `128`            | Batch size for training                                             |
| `--population`   | int   | `20`             | Population size for evolutionary algorithms                         |
| `--iterations`   | int   | `15`             | Number of search iterations                                         |
| `--sub-train`    | int   | `20000`          | Number of training samples for optimization                         |
| `--sub-test`     | int   | `5000`           | Number of test samples for optimization                             |
| `--output-dir`   | str   | `./results`      | Directory to save results                                           |
| `--seed`         | int   | `42`             | Random seed for reproducibility                                     |
| `--data`         | str   | `mnist`          | Dataset to use (`mnist`, `cifar`)                                   |
| `--input-dim`    | tuple | `(1, 28, 28)`    | Input dimensions of the data                                        |

Genetic Algorithm Specific Parameters

| Argument            | Type  | Default | Description                   |
| ------------------- | ----- | ------- | ----------------------------- |
| `--mutation-rate`   | float | `0.1`   | Mutation rate                 |
| `--tournament-size` | int   | `5`     | Tournament size for selection |
| `--crossover-prob`  | float | `0.75`  | Crossover probability         |

Firefly Algorithm Specific Parameters

| Argument   | Type  | Default | Description                                |
| ---------- | ----- | ------- | ------------------------------------------ |
| `--alpha`  | float | `0.5`   | Alpha parameter                            |
| `--beta0`  | float | `1.0`   | Beta0 parameter                            |
| `--gamma`  | float | `1.0`   | Gamma parameter                            |
| `--sigma0` | float | `1.0`   | Standard deviation for normal distribution |
| `--prob`   | float | `0.5`   | Probability to use normal distribution     |

### Configuration File (`config.json`)

You can define fixed and variable parameters for your search space in the `config.json` file. This includes:

- **Filter sizes**: e.g., `[3, 5, 7, 9]`

- **Number of filters**: e.g., `[16, 32, 64, 128]`

- **Strides**: e.g., `[1, 2, 3, 4]`

- **Activation functions**: e.g., `["relu", "tanh", "sigmoid", "leaky_relu"]`

- **Number of neurons in fully connected layers**: e.g., `[8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 512]`

- **Fixed parameters**: padding, maximum and minimum number of layers, etc.

This allows for a fully configurable yet controlled search space.

---

## License

This project is licensed under the MIT License.
