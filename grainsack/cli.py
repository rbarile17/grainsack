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

from grainsack import KGS, MODELS
from grainsack.evaluate import run_evaluate
from grainsack.explain import build_combinatorial_optimization_explainer, run_explain
from grainsack.kg import KG
from grainsack.lp import MODEL_REGISTRY
from grainsack.utils import load_kge_model, read_json, write_json
from grainsack.workflow import Comparison, Validation


def set_seeds(seed):
    """Set the random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())


@click.group()
def cli():
    """Unified grainsack CLI."""


@cli.command()
@click.option("--kg_name", type=click.Choice(KGS))
@click.option("--kge_model_name", type=click.Choice(MODELS))
@click.option("--output_path", type=click.Path(), default="kge_config.json")
def tune(kg_name, kge_model_name, output_path):
    """Select for the given KGE model the best hyperparameter config based on the performance on the given KG."""
    set_seeds(42)

    print(f"Loading KG...")
    kg = KG(kg_name)

    num_epochs = MODEL_REGISTRY[kge_model_name]["epochs"]
    batch_size = MODEL_REGISTRY[kge_model_name]["batch_size"]

    config = hpo_pipeline(
        timeout=1,
        training=kg.training,
        validation=kg.validation,
        testing=kg.testing,
        model=kge_model_name,
        training_kwargs={"num_epochs": num_epochs, "batch_size": batch_size},
        stopper="early",
        stopper_kwargs={"frequency": 5, "patience": 2, "relative_delta": 0.002},
        lr_scheduler="ExponentialLR",
    )

    config = config._get_best_study_config()["pipeline"]

    config["model"] = kge_model_name

    config.pop("training")
    config.pop("validation")
    config.pop("testing")

    write_json(config, output_path)


@cli.command()
@click.option("--kg_name", type=click.Choice(KGS))
@click.option("--kge_config_path", type=click.Path(exists=True))
@click.option("--output_path", type=click.Path(), default="kge_model.pt")
def train(kg_name, kge_config_path, output_path):
    """Train on the given KG the KGE model specified in the given config along with the hyperparameters."""
    set_seeds(42)

    config = read_json(kge_config_path)
    print(f"Loading KG...")
    kg = KG(kg_name)

    result = pipeline(
        **config,
        training=kg.training,
        testing=kg.testing,
        validation=kg.validation,
        stopper="early",
        stopper_kwargs={"frequency": 5, "patience": 2, "relative_delta": 0.002},
        evaluation_kwargs={"batch_size": 64},
    )

    torch.save(result.model.state_dict(), output_path)


@cli.command()
@click.option("--kg_name", type=click.Choice(KGS))
@click.option("--kge_model_path", type=click.Path(exists=True))
@click.option("--kge_config_path", type=click.Path(exists=True))
@click.option("--output_path", type=click.Path(), default="predictions.csv")
def rank(kg_name, kge_model_path, kge_config_path, output_path):
    """Compute the rank of each triple in the given KG via the given KGE model."""
    set_seeds(42)

    print(f"Loading KG...")
    kg = KG(kg_name)

    kge_config = read_json(kge_config_path)

    kge_model = load_kge_model(kge_model_path, kge_config, kg)
    kge_model.eval()
    kge_model.cuda()

    evaluator = RankBasedEvaluator(clear_on_finalize=False)
    mapped_triples = kg.testing.mapped_triples
    mapped_triples = mapped_triples.cuda()
    filter_triples = [kg.training.mapped_triples.cuda(), kg.validation.mapped_triples.cuda()]
    evaluator.evaluate(kge_model, mapped_triples, batch_size=64, additional_filter_triples=filter_triples)

    ranks = evaluator.ranks[("tail", "optimistic")]
    ranks = np.concatenate(ranks)
    output = np.concatenate((kg.testing.triples, ranks.reshape(-1, 1)), axis=1)
    output = pd.DataFrame(output, columns=["s", "p", "o", "rank"])
    output.to_csv(output_path, index=False, sep=";")


@cli.command()
@click.option("--predictions_path", type=click.Path(exists=True))
@click.option("--output_path", type=click.Path(), default="select_predictions.csv")
def select_predictions(predictions_path, output_path):
    """Select the top 5 predictions for each entity."""
    set_seeds(42)

    predictions = pd.read_csv(predictions_path, sep=";")

    predictions = predictions[predictions["rank"] == 1]
    predictions.drop(["rank"], axis=1, inplace=True)

    predictions = predictions.sample(5)
    predictions = predictions.reset_index(drop=True)

    predictions.to_csv(output_path, sep="\t", index=False, header=False)


@cli.command()
@click.option("--predictions_path", type=click.Path(exists=True))
@click.option("--kg_name", type=click.Choice(KGS))
@click.option("--kge_model_path", type=click.Path(exists=True))
@click.option("--kge_config_path", type=click.Path(exists=True))
@click.option("--lpx_config", type=str)
@click.option("--output_path", type=click.Path(), default="explanations.json")
def explain(predictions_path, kg_name, kge_model_path, kge_config_path, lpx_config, output_path):
    """Compute the explanations for the given predictions based on the given data, models, and config.

    Compute the explanations for the given KG (predictions) using the statements in the other given KG, based on the given KGE model and the associated hyperparameter config (used for making the predictions), and according to the given explanation config.
    """
    set_seeds(42)

    lpx_config = json.loads(lpx_config)

    print(f"Loading KG...")
    kg = KG(kg=kg_name)

    print("Loading predictions...")
    with open(predictions_path, "r", encoding="utf-8") as predictions:
        predictions = [x.strip().split("\t") for x in predictions.readlines()]
    predictions = kg.id_triples(predictions)

    kge_config = read_json(kge_config_path)
    # lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]

    explanations = run_explain(
        predictions, kg, kge_model_path, kge_config, lpx_config, build_combinatorial_optimization_explainer
    )

    output = []
    for i in range(len(predictions)):
        output.append(
            {
                "prediction": kg.label_triple(predictions[i]),
                "explanation": kg.label_triples([tuple(row.tolist()) for row in explanations[i]]),
            }
        )

    write_json(output, output_path)


@cli.command()
@click.option("--explanations_path", type=click.Path(exists=True))
@click.option("--kg_name", type=click.Choice(KGS))
@click.option("--kge_model_path", type=click.Path(exists=True))
@click.option("--kge_config_path", type=click.Path(exists=True))
@click.option("--eval_config", type=str)
@click.option("--output_path", type=click.Path(), default="evaluations.json")
def evaluate(explanations_path, kg_name, kge_model_path, kge_config_path, eval_config, output_path):
    """Evaluates the given explanations for the given KG (predictions) according to the given evaluation config."""
    set_seeds(42)

    print(f"Loading explanations...")
    explained_predictions = read_json(explanations_path)

    print(f"Loading KG...")
    kg = KG(kg=kg_name)

    eval_config = json.loads(eval_config)
    evalautions = run_evaluate(explained_predictions, kg, kge_model_path, kge_config_path, eval_config)

    write_json(evalautions, output_path)


@cli.command()
def validate():
    """Validate the pipeline."""
    luigi.build([Validation()], local_scheduler=True)


@cli.command()
def comparison():
    """Run the comparison task."""
    luigi.build([Comparison()], local_scheduler=True)


if __name__ == "__main__":
    cli()
