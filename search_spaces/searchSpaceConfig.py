import torch.nn as nn
import json
import os
import yaml
from typing import List, Type, Any, Dict, Union
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # Default values
    FILTERS_MAP = [1, 8, 16, 32]
    KERNEL_MAP = [3, 5, 7, 9]
    STRIDE_MAP = [1, 2, 3, 4]
    CONV_PADDING = 1
    POOL_PADDING = 1
    ACTIVATION_FUNCTIONS = [nn.ReLU, nn.ELU, nn.Sigmoid, nn.LeakyReLU]
    FC_SIZES = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 512]
    MAX_LAYERS = 15
    MIN_LAYERS = 2
    CROMOSOME_SIZE = 8
    
    # List of configurable attributes
    _CONFIGURABLE_ATTRS = [
        'FILTERS_MAP', 'KERNEL_MAP', 'STRIDE_MAP', 'CONV_PADDING', 
        'POOL_PADDING', 'FC_SIZES', 'MAX_LAYERS', 'MIN_LAYERS', 'CROMOSOME_SIZE'
    ]
    
    # Mapping for activation functions (not directly editable via file)
    _ACTIVATION_MAPPING = {
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'LeakyReLU': nn.LeakyReLU,
        'SELU': nn.SELU,
        'GELU': nn.GELU
    }
    
    @classmethod
    def load_config(cls, config_path: str = None) -> None:
        """
            Loads the configuration from a JSON or YAML file.
            If no file is specified, searches for a config.json or config.yaml file
            in the current directory or a 'config' directory.
        """
        if config_path is None:
            
            possible_paths = [
                'config.json', 'config.yaml', 'config.yml',
                os.path.join('config', 'config.json'),
                os.path.join('config', 'config.yaml'),
                os.path.join('config', 'config.yml')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"No configuration file found. Using default values.")
            return
        
        try:
            if config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:  
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            
            # Updating attributes
            cls._update_from_dict(config_data)
            logger.info(f"Configuration successfully loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    @classmethod
    def _update_from_dict(cls, config_data: Dict[str, Any]) -> None:
        """Updates class attributes from a dictionary."""
        for key, value in config_data.items():
            if key in cls._CONFIGURABLE_ATTRS:
                setattr(cls, key, value)
                logger.debug(f"Attribut {key} updated : {value}")
            elif key == 'ACTIVATION_FUNCTIONS':
                if isinstance(value, list):
                    activation_classes = []
                    for func_name in value:
                        if func_name in cls._ACTIVATION_MAPPING:
                            activation_classes.append(cls._ACTIVATION_MAPPING[func_name])
                        else:
                            logger.warning(f"Unknown activation function: {func_name}")
                    
                    if activation_classes:
                        setattr(cls, 'ACTIVATION_FUNCTIONS', activation_classes)
            else:
                logger.warning(f"Unknown attribute in configuration: {key}")
    
    @classmethod
    def save_current_config(cls, output_path: str = 'config.json') -> None:
        """Saves the current configuration to a file."""
        config_data = {}
        
        for attr in cls._CONFIGURABLE_ATTRS:
            config_data[attr] = getattr(cls, attr)
        
        activation_names = []
        for func in cls.ACTIVATION_FUNCTIONS:
            func_name = func.__name__
            if func_name in cls._ACTIVATION_MAPPING.values():
                for name, cls_obj in cls._ACTIVATION_MAPPING.items():
                    if cls_obj == func:
                        activation_names.append(name)
                        break
        
        config_data['ACTIVATION_FUNCTIONS'] = activation_names
        
        try:
            if output_path.endswith(('.yaml', '.yml')):
                with open(output_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:  
                with open(output_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
            
            logger.info(f"Configuration successfully saved in {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def get_config_as_dict(cls) -> Dict[str, Any]:
        """Returns the current configuration as a dictionary."""
        config_dict = {}
        for attr in cls._CONFIGURABLE_ATTRS:
            config_dict[attr] = getattr(cls, attr)
        
        activation_names = []
        for func in cls.ACTIVATION_FUNCTIONS:
            for name, cls_obj in cls._ACTIVATION_MAPPING.items():
                if cls_obj == func:
                    activation_names.append(name)
                    break
        
        config_dict['ACTIVATION_FUNCTIONS'] = activation_names
        return config_dict



if __name__ == "__main__":
    print("Default configuration:", Config.FILTERS_MAP)
    
    Config.load_config()
    
    print("Configuration after loading:", Config.FILTERS_MAP)
    
    Config.save_current_config('my_config.json')