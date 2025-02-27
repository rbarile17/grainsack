"""Utility functions for loading and saving models and data."""


import json

import torch

from .kge_lp import MODEL_REGISTRY


def init_kge_model(config, kg):
    """Initialize the model with the given configuration and knowledge graph."""
    kge_model_name = config["model"]
    kge_model_class = MODEL_REGISTRY[kge_model_name]["class"]
    kge_model = kge_model_class(triples_factory=kg.training, **config["model_kwargs"], random_seed=42)

    return kge_model


def load_kge_model(kge_model_path, kge_config, kg):
    """Load the model from the given configuration and knowledge graph."""
    kge_model = init_kge_model(kge_config, kg)
    kge_model.load_state_dict(torch.load(kge_model_path, weights_only=True))

    return kge_model

def read_json(path):
    """Read a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
