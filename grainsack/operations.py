"""This module defines the command-line interface (CLI) for the grainsack package."""

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

from grainsack import GROUND_TRUTH
from grainsack.evaluate import run_evaluate
from grainsack.explain import run_explain
from grainsack.kg import KG
from grainsack.kge_lp import MODEL_REGISTRY
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


@cli.command(
    help=(
        "Select for the KGE model specified in the given config the best hyperparameter config based on the performance on the given KG. "
        "Save the config to a .json file."
    )
)
@click.option("--kg_name", help="The name of the KG.")
@click.option("--kge_model_name", help="The name of the KGE model to tune.")
@click.option(
    "--output_path", type=click.Path(), default="kge_config.json", help="The path to save the best hyperparameter config."
)
def tune(kg_name, kge_model_name, output_path):
    """Perform hyperparameter tuning.

    Select for the KGE model specified in the given config the best hyperparameter config based on the performance on the given KG.
    Save the config to a .json file.

    :param kg_name: The name of the KG to be used for assessing the performance of the possible hyperparameter configs.
    :type kg_name: str
    :param kge_model_name: The name of the KGE model to select the hyperparameter config for.
    :type kge_model_name: str
    :param output_path: The path to save the best hyperparameter config.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
    """
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


@cli.command(
    help=(
        "Train on the given KG the KGE model specified in the given config along with the hyperparameters. "
        "Save the model parameters to a .pt file."
    )
)
@click.option("--kg_name", help="The name of the KG to train on.")
@click.option("--kge_config_path", type=click.Path(exists=True), help="The path to the KGE config file.")
@click.option("--output_path", type=click.Path(), default="kge_model.pt", help="The path to save the trained KGE model.")
def train(kg_name, kge_config_path, output_path):
    """Train on the given KG the given KGE model.

    Train on the given KG the KGE model specified in the given config along with the hyperparameters.
    Save the model parameters to a .pt file.

    :param kg_name: The name of the KG to train on.
    :type kg_name: str
    :param kge_config_path: The path to the KGE config file.
    :type kge_config_path: pathlib.Path
    :param output_path: The path to save the trained KGE model.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
    """
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


@cli.command(
    help="Compute the rank of each triple in the given KG via the given KGE model. Save the ranked triples to a .csv file."
)
@click.option("--kg_name", help="The name of the KG to compute the ranks for.")
@click.option("--kge_model_path", type=click.Path(exists=True), help="The path to the KGE .pt model file.")
@click.option("--kge_config_path", type=click.Path(exists=True), help="The path to the KGE .json config file.")
@click.option("--output_path", type=click.Path(), default="predictions.csv", help="The path to save the predictions with ranks.")
def rank(kg_name, kge_model_path, kge_config_path, output_path):
    """Compute the rank of the given triples.

    Compute the rank of each triple in the given KG via the given KGE model instantiated according to the given KGE config.
    Save the ranked predictions to a .csv file.

    :param kg_name: The name of the KG to compute the ranks for.
    :type kg_name: str
    :param kge_model_path: The path to the KGE .pt model file.
    :type kge_model_path: pathlib.Path
    :param kge_config_path: The path to the KGE .json config file.
    :type kge_config_path: pathlib.Path
    :param output_path: The path to save the predictions with ranks.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
    """
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


@cli.command(help="Select the top ranked triples from the given triples. Save the selected triples to a .csv file.")
@click.option("--predictions_path", type=click.Path(exists=True), help="The path to the ranked triples.")
@click.option("--output_path", type=click.Path(), default="select_predictions.csv", help="The path to save the selected triples.")
def select_predictions(predictions_path, output_path):
    """Select the top ranked triples from the given triples.

    Save the selected triples to a .csv file.

    :param predictions_path: The path to the ranked triples.
    :type predictions_path: pathlib.Path
    :param output_path: The path to save the selected triples.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
    """
    set_seeds(42)

    predictions = pd.read_csv(predictions_path, sep=";")

    predictions = predictions[predictions["rank"] == 1]
    predictions.drop(["rank"], axis=1, inplace=True)

    predictions = predictions.sample(5)
    predictions = predictions.reset_index(drop=True)

    predictions.to_csv(output_path, sep="\t", index=False, header=False)


@cli.command(
    help=(
        "Compute the explanations for the given KG (predictions) using the statements in the other given KG, "
        "based on the given KGE model and the associated hyperparameter config (used for making the predictions), and according to the given explanation config."
        "Save the explanations to a .json file."
    )
)
@click.option("--predictions_path", type=click.Path(exists=True), help="The path to the predictions to be explained.")
@click.option("--kg_name", type=str, help="The name of the KG to be used for computing the explanations.")
@click.option(
    "--kge_model_path", type=click.Path(exists=True), help="The path to the KGE .pt model file used for the predictions."
)
@click.option(
    "--kge_config_path", type=click.Path(exists=True), help="The path to the KGE .json config file used for the predictions."
)
@click.option(
    "--lpx_config",
    type=str,
    help="The explanation config as a JSON string. It should contain the method and the parameters for the explanation method.",
)
@click.option(
    "--factory_name",
    type=str,
    help="The name of the function building the explanation function from the explanation config.",
    default="build_combinatorial_optimization_explainer"
)
@click.option("--output_path", type=click.Path(), default="explanations.json", help="The path to save the explanations.")
def explain(predictions_path, kg_name, kge_model_path, kge_config_path, lpx_config, output_path, factory_name):
    """Compute the explanations for the given predictions based on the given data, models, and config.

    Compute the explanations for the given KG (predictions) using the statements in the other given KG,
    based on the given KGE model and the associated hyperparameter config (used for making the predictions), and according to the given explanation config.
    Save the explanations to a .json file.

    :param predictions_path: The path to the predictions to be explained.
    :type predictions_path: pathlib.Path
    :param kg_name: The name of the KG to be used for computing the explanations.
    :type kg_name: str
    :param kge_model_path: The path to the KGE .pt model file used for the predictions.
    :type kge_model_path: pathlib.Path
    :param kge_config_path: The path to the KGE .json config file used for the predictions.
    :type kge_config_path: pathlib.Path
    :param lpx_config: The explanation config as a JSON string. It should contain the method and the parameters for the explanation method.
    :type lpx_config: str
    :param output_path: The path to save the explanations.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
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

    factory = globals().get(factory_name)

    explanations = run_explain(
        predictions, kg, kge_model_path, kge_config, lpx_config, factory
    )

    output = []

    if lpx_config["method"] != GROUND_TRUTH:
        for i in range(len(predictions)):
            output.append(
                {
                    "prediction": kg.label_triple(torch.tensor(predictions[i])),
                    "explanation": kg.label_triples(explanations[i]),
                }
            )

        write_json(output, output_path)
    else:
        write_json(explanations, output_path)


@cli.command(
    help=(
        "Evaluate the given explanations (each associated to a prediction) according to the given"
        "evaluation config and possibly adopting the given KGE model and associated config."
    )
)
@click.option(
    "--explanations_path",
    type=click.Path(exists=True),
    help="The path to the explanations (each associated to a prediction) to be evaluated.",
)
@click.option("--kg_name", help="The name of the KG used for the explanations and to be used for the evaluation.")
@click.option(
    "--kge_model_path",
    type=click.Path(exists=True),
    help="The path to the KGE .pt model file used for the explanations and to be possibly used (depending on the prompting method) for the evaluation.",
)
@click.option(
    "--kge_config_path",
    type=click.Path(exists=True),
    help="The path to the KGE .json config file used for the explanations and to be possibly used (depending on the prompting method) for the evaluation.",
)
@click.option(
    "--eval_config",
    type=str,
    help="The evaluation config as a JSON string. It should contain the LLM id (from unsloth) and the prompting method (zero_shot, zero_shot_constrained, few_shot, few_shot_constrained).",
)
@click.option("--output_path", type=click.Path(), default="evaluations.json", help="The path to save the evaluations.")
def evaluate(explanations_path, kg_name, kge_model_path, kge_config_path, eval_config, output_path):
    """Evaluate the given explanations based on the given data, models, and config.

    Evaluate the given explanations (each associated to a prediction) according to the given evaluation config and possibly adopting the given KGE model and associated config.

    :param explanations_path: The path to the explanations (each associated to a prediction) to be evaluated.
    :type explanations_path: pathlib.Path
    :param kg_name: The name of the KG used for the explanations and to be used for the evaluation.
    :type kg_name: str
    :param kge_model_path: The path to the KGE .pt model file used for the explanations and 
    to be possibly used (depending on the prompting method) for the evaluation.
    :type kge_model_path: pathlib.Path
    :param kge_config_path: The path to the KGE .json config file used for the explanations and
    to be possibly used (depending on the prompting method) for the evaluation.
    :type kge_config_path: pathlib.Path
    :param eval_config: The evaluation config as a JSON string. 
    It should contain the LLM id (from unsloth) and the prompting method (zero_shot, zero_shot_constrained, few_shot, few_shot_constrained).
    :type eval_config: str
    :param output_path: The path to save the evaluations.
    :type output_path: pathlib.Path
    :return: None
    :rtype: None
    """
    set_seeds(42)

    print(f"Loading explanations...")
    explained_predictions = read_json(explanations_path)

    print(f"Loading KG...")
    kg = KG(kg=kg_name)

    eval_config = json.loads(eval_config)
    evalautions = run_evaluate(explained_predictions, kg, kge_model_path, kge_config_path, eval_config)

    write_json(evalautions, output_path)


@cli.command()
def validation():
    """Run the validation workflow adopting the setup in validation_setup.csv."""
    luigi.build([Validation()], local_scheduler=True)


@cli.command()
def comparison():
    """Run the comparison workflow adopting the setup in comparison_setup.csv."""
    luigi.build([Comparison()], local_scheduler=True)


if __name__ == "__main__":
    cli()
