import json
import random

import click
import luigi
import numpy as np
import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline

from grainsack import KGES_PATH, KGS, LP_CONFIGS_PATH, METHODS, MODELS, PREDICTIONS_PATH
from grainsack.dixit import run_dixit_task
from grainsack.explain import (
    build_combinatorial_optimization_explainer,
    run_explanation_task,
)
from grainsack.kg import KG
from grainsack.lp import MODEL_REGISTRY
from grainsack.utils import load_model, read_json, write_json
from grainsack.workflow import Validation


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())


@click.group()
def cli():
    """Unified grainsack CLI."""
    pass


@cli.command()
@click.option("--model", type=click.Choice(MODELS))
@click.option("--kg", type=click.Choice(KGS))
def train(model: str, kg: str) -> None:
    set_seeds(42)

    config = LP_CONFIGS_PATH / f"{model}_{kg}.json"
    config = read_json(config)
    print(f"Loading dataset {kg}...")
    kg = KG(kg=kg)

    result = pipeline(
        **config,
        training=kg.training,
        testing=kg.testing,
        validation=kg.validation,
        stopper="early",
        stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
    )

    torch.save(result.model.state_dict(), KGES_PATH / f"{model}_{kg.name}.pt")


@cli.command()
@click.option("--kg", type=click.Choice(KGS))
@click.option("--model", type=click.Choice(MODELS))
def tune(kg: str, model: str) -> None:
    model_name = model

    set_seeds(42)

    print(f"Loading dataset {kg}...")
    kg = KG(kg)

    num_epochs = MODEL_REGISTRY[model]["epochs"]
    batch_size = MODEL_REGISTRY[model]["batch_size"]

    config = hpo_pipeline(
        timeout=1,
        training=kg.training,
        validation=kg.validation,
        testing=kg.testing,
        model=model,
        training_kwargs=dict(num_epochs=num_epochs, batch_size=batch_size),
        stopper="early",
        stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
        lr_scheduler="ExponentialLR",
    )

    config = config._get_best_study_config()["pipeline"]

    config["model"] = model_name

    config.pop("training")
    config.pop("validation")
    config.pop("testing")

    write_json(config, LP_CONFIGS_PATH / f"{model}_{kg.name}.json")


@cli.command()
@click.option("--kg", type=click.Choice(KGS))
@click.option("--model", type=click.Choice(MODELS))
def predict(kg: str, model: str) -> None:
    set_seeds(42)

    predictions_path = PREDICTIONS_PATH / f"{model}_{kg}.csv"

    config = LP_CONFIGS_PATH / f"{model}_{kg}.json"
    config = read_json(config)

    print(f"Loading {kg}...")
    kg = KG(kg=kg)

    model = load_model(config, kg)
    model.eval()
    model.cuda()

    evaluator = RankBasedEvaluator(clear_on_finalize=False)
    mapped_triples = kg.testing.mapped_triples
    mapped_triples = mapped_triples.cuda()
    filter_triples = [kg.training.mapped_triples.cuda(), kg.validation.mapped_triples.cuda()]
    evaluator.evaluate(model, mapped_triples, batch_size=64, additional_filter_triples=filter_triples)

    ranks = evaluator.ranks[("tail", "optimistic")]
    ranks = np.concatenate(ranks)
    output = np.concatenate((kg.testing.triples, ranks.reshape(-1, 1)), axis=1)
    output = pd.DataFrame(output, columns=["s", "p", "o", "rank"])
    output.to_csv(predictions_path, index=False, sep=";")


@cli.command()
@click.option("--kg", type=click.Choice(KGS))
@click.option("--model", type=click.Choice(MODELS))
def select_predictions(kg: str, model: str) -> None:
    set_seeds(42)

    preds_path = PREDICTIONS_PATH / f"{model}_{kg}.csv"
    preds_path2 = PREDICTIONS_PATH / f"{model}_{kg}2.csv"

    preds = pd.read_csv(preds_path, sep=";")

    # preds = preds[preds["rank"] == 1]
    preds.drop(["rank"], axis=1, inplace=True)

    preds = preds.sample(5)
    preds = preds.reset_index(drop=True)

    preds.to_csv(preds_path2, sep="\t", index=False, header=False)


@cli.command()
@click.option("--kg", type=click.Choice(KGS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS))
@click.option("--explanation_kwargs", type=str)
def explain(model, kg, method, explanation_kwargs) -> None:
    set_seeds(42)
    explanation_kwargs = json.loads(explanation_kwargs)
    run_explanation_task(model, kg, method, explanation_kwargs, build_combinatorial_optimization_explainer)


@cli.command()
@click.option("--kg", type=click.Choice(KGS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS))
@click.option("--explanation_kwargs", type=str)
@click.option("--llm")
def dixit(model, kg, method, explanation_kwargs, llm):
    set_seeds(42)
    explanation_kwargs = json.loads(explanation_kwargs)
    run_dixit_task(model, kg, method, explanation_kwargs, llm)


@cli.command()
def validate():
    luigi.build([Validation()], local_scheduler=True)

@cli.command()
def comparison():
    luigi.build([Comparison()], local_scheduler=True)

if __name__ == "__main__":
    cli()
