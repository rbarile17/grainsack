"""Utility functions for loading and saving models and data."""


import json

import torch

from . import KGES_PATH
from .lp import MODEL_REGISTRY


def init_model(config, kg):
    """Initialize the model with the given configuration and knowledge graph."""
    model = config["model"]
    model_class = MODEL_REGISTRY[model]["class"]
    model = model_class(triples_factory=kg.training, **config["model_kwargs"], random_seed=42)

    return model


def load_model(config, kg):
    """Load the model from the given configuration and knowledge graph."""
    model = config["model"]
    model_path = KGES_PATH / f"{model}_{kg.name}.pt"
    model = init_model(config, kg)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model

def read_json(path):
    """Read a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
