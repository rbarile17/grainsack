"""Utility functions for loading and saving models and data."""

import json

import torch

from .kge_lp import MODEL_REGISTRY


def init_kge_model(config, kg):
    """Initialize a KGE model with the given configuration and knowledge graph.
    
    Args:
        config (dict): Configuration dictionary containing 'model' name, 'model_kwargs',
            and optional 'loss_kwargs'.
        kg: Knowledge graph object with training triples factory.
        
    Returns:
        KGE model instance initialized with the specified configuration.
    """
    kge_model_name = config["model"]
    kge_model_class = MODEL_REGISTRY[kge_model_name]["class"]
    kge_model = kge_model_class(
        triples_factory=kg.training, **config["model_kwargs"], loss_kwargs=config.get("loss_kwargs", {}), random_seed=42
    )

    return kge_model


def load_kge_model(kge_model_path, kge_config, kg):
    """Load a pre-trained KGE model from disk.
    
    Args:
        kge_model_path (str or Path): Path to the saved model state dictionary (.pt file).
        kge_config (dict): Configuration dictionary for initializing the model.
        kg: Knowledge graph object with training triples factory.
        
    Returns:
        Loaded KGE model with restored weights.
    """
    kge_model = init_kge_model(kge_config, kg)
    kge_model.load_state_dict(torch.load(kge_model_path, weights_only=True))

    return kge_model


def read_json(path):
    """Read and parse a JSON file.
    
    Args:
        path (str or Path): Path to the JSON file to read.
        
    Returns:
        dict or list: Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """Write data to a JSON file with indentation.
    
    Args:
        data (dict or list): Data to serialize to JSON.
        path (str or Path): Path where the JSON file will be saved.
        
    Returns:
        None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
