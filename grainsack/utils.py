import json

import torch

from . import KGES_PATH
from .lp import MODEL_REGISTRY


def init_model(config, kg):
    model = config["model"]
    model_class = MODEL_REGISTRY[model]["class"]
    model = model_class(triples_factory=kg.training, **config["model_kwargs"], random_seed=42)

    return model


def load_model(config, kg):
    model = config["model"]
    model_path = KGES_PATH / f"{model}_{kg.name}.pt"
    model = init_model(config, kg)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

