"""Command-line interface for the GRAINSACK knowledge graph explanation benchmark.

Provides commands for the complete KGE model and explanation workflow:
- tune: Hyperparameter optimization for KGE models
- train: Model training with early stopping
- rank: Prediction ranking on test triples
- select_predictions: Sample top predictions for explanation
- explain: Generate explanations using various methods
- evaluate: Evaluate explanation quality
- comparison: Run complete benchmarking workflow
"""

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

from grainsack import logger
from grainsack.evaluate import run_evaluate
from grainsack.explain import build_combinatorial_optimization_explainer, run_explain
from grainsack.kg import KG
from grainsack.kge_lp import MODEL_REGISTRY
from grainsack.mhs_explain import mhs_explain_factory
from grainsack.utils import load_kge_model, read_json, write_json
from grainsack.workflow import Comparison

HELP_KG_NAME = "Knowledge graph name (e.g., 'ARCO25-5-MATERIALIZE', 'ATRAVEL-MATERIALIZE')."
HELP_KGE_MODEL_PATH = "Path to trained KGE model weights (.pt file)."
HELP_KGE_CONFIG_PATH = "Path to KGE model configuration (.json file with hyperparameters)."
HELP_OUTPUT_PATH = "Path where output file will be saved."


def set_seeds(seed):
    """Set random seeds for reproducible results across libraries.
    
    Configures random number generators for NumPy, PyTorch (CPU and CUDA),
    and Python's random module to ensure deterministic behavior.
    
    Args:
        seed (int): Random seed value to use across all libraries.
    """
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())


@click.group()
def cli():
    """GRAINSACK CLI entry point."""


@cli.command(
    help="Perform hyperparameter optimization for a KGE model on a knowledge graph."
)
@click.option("--kg_name", required=True, help=HELP_KG_NAME)
@click.option("--kge_model_name", required=True, help="KGE model architecture (e.g., 'TransE', 'ComplEx', 'ConvE').")
@click.option(
    "--output_path", type=click.Path(), default="kge_config.json", help="Path to save optimized hyperparameter configuration."
)
def tune(kg_name, kge_model_name, output_path):
    """Hyperparameter optimization for KGE models."""
    set_seeds(42)
    
    logger.info(f"Starting hyperparameter tuning for {kge_model_name} on {kg_name}")

    kg = KG(kg_name, create_inverse_triples=kge_model_name == "ConvE")

    num_epochs = MODEL_REGISTRY[kge_model_name]["epochs"]
    batch_size = MODEL_REGISTRY[kge_model_name]["batch_size"]

    config = hpo_pipeline(
        timeout=8 * 60 * 60 if kge_model_name != "TransE" else 4 * 60 * 60, 
        training=kg.training,
        validation=kg.validation,
        testing=kg.testing,
        model=kge_model_name,
        training_kwargs={"num_epochs": num_epochs, "batch_size": batch_size},
        stopper="early",
        stopper_kwargs={"frequency": 5, "patience": 2, "relative_delta": 0.002},
        lr_scheduler="ExponentialLR",
        evaluation_kwargs={"batch_size": 16356},
    )

    config = config._get_best_study_config()["pipeline"]

    config["model"] = kge_model_name

    config.pop("training")
    config.pop("validation")
    config.pop("testing")

    write_json(config, output_path)
    logger.info(f"Hyperparameter tuning completed. Config saved to {output_path}")


@cli.command(
    help="Train a KGE model on a knowledge graph with early stopping."
)
@click.option("--kg_name", required=True, help=HELP_KG_NAME)
@click.option("--kge_config_path", type=click.Path(exists=True), required=True, help=HELP_KGE_CONFIG_PATH)
@click.option("--output_path", type=click.Path(), default="kge_model.pt", help="Path to save trained model weights.")
def train(kg_name, kge_config_path, output_path):
    """Train KGE model with early stopping."""
    set_seeds(42)
    
    logger.info(f"Starting training for {kg_name}")

    config = read_json(kge_config_path)
    kg = KG(kg_name, create_inverse_triples=config["model"] == "ConvE")

    result = pipeline(
        **config,
        training=kg.training,
        testing=kg.testing,
        validation=kg.validation,
        stopper="early",
        stopper_kwargs={"frequency": 5, "patience": 2, "relative_delta": 0.002, "evaluation_batch_size": 16356},
    )

    config["training_kwargs"]["num_epochs"] = result.stopper.best_epoch
    write_json(config, kge_config_path)

    torch.save(result.model.state_dict(), output_path)
    logger.info(f"Training completed. Model saved to {output_path}")


@cli.command(
    help="Rank test triples using a trained KGE model (filtered setting)."
)
@click.option("--kg_name", required=True, help=HELP_KG_NAME)
@click.option("--kge_model_path", type=click.Path(exists=True), required=True, help=HELP_KGE_MODEL_PATH)
@click.option("--kge_config_path", type=click.Path(exists=True), required=True, help=HELP_KGE_CONFIG_PATH)
@click.option("--output_path", type=click.Path(), default="predictions.csv", help="Path to save ranked predictions CSV (columns: s, p, o, rank).")
def rank(kg_name, kge_model_path, kge_config_path, output_path):
    """Rank test triples using trained KGE model (filtered setting)."""
    set_seeds(42)
    
    logger.info(f"Starting ranking for {kg_name}")

    kge_config = read_json(kge_config_path)

    kg = KG(kg_name, create_inverse_triples=kge_config["model"] == "ConvE")

    kge_model = load_kge_model(kge_model_path, kge_config, kg)
    kge_model.eval()
    kge_model.cuda()

    evaluator = RankBasedEvaluator(clear_on_finalize=False)
    mapped_triples = kg.testing.mapped_triples.cuda()
    filter_triples = [kg.training.mapped_triples.cuda(), kg.validation.mapped_triples.cuda()]
    evaluator.evaluate(kge_model, mapped_triples, batch_size=16356, additional_filter_triples=filter_triples)

    ranks = evaluator.ranks[("tail", "optimistic")]
    ranks = np.concatenate(ranks)
    output = np.concatenate((kg.testing.triples, ranks.reshape(-1, 1)), axis=1)
    output = pd.DataFrame(output, columns=["s", "p", "o", "rank"])
    output.to_csv(output_path, index=False, sep=";")
    logger.info(f"Ranking completed. Predictions saved to {output_path}")


@cli.command(help="Sample top-ranked predictions (rank=1) for explanation experiments.")
@click.option("--predictions_path", type=click.Path(exists=True), required=True, help="Path to CSV file with ranked predictions.")
@click.option(
    "--output_path", type=click.Path(), default="selected_predictions.csv", help="Path to save sampled predictions (up to 100 triples)."
)
def select_predictions(predictions_path, output_path):
    """Sample top-ranked predictions (rank=1) for explanation."""
    logger.info(f"Selecting top predictions from {predictions_path}")
    
    predictions = pd.read_csv(predictions_path, sep=";")

    predictions = predictions[predictions["rank"] == 1]
    predictions.drop(["rank"], axis=1, inplace=True)

    sample_size = min(100, len(predictions))
    predictions = predictions.sample(sample_size, random_state=42)
    predictions = predictions.reset_index(drop=True)

    predictions.to_csv(output_path, sep="\t", index=False, header=False)
    logger.info(f"Selected {sample_size} predictions and saved to {output_path}")


@cli.command(
    help="Generate explanations for link predictions using various explanation methods."
)
@click.option("--predictions_path", type=click.Path(exists=True), required=True, help="Path to TSV file with predictions to explain (tab-separated: s p o).")
@click.option("--kg_name", type=str, required=True, help=HELP_KG_NAME)
@click.option(
    "--kge_model_path", type=click.Path(exists=True), required=True, help=HELP_KGE_MODEL_PATH
)
@click.option(
    "--kge_config_path", type=click.Path(exists=True), required=True, help=HELP_KGE_CONFIG_PATH
)
@click.option(
    "--lpx_config",
    type=str,
    required=True,
    help="Explanation method configuration as JSON string (must include 'method' key: KELPIE, IMAGINE, CRIAGE, DATA_POISONING, mhs_explain).",
)
@click.option(
    "--factory_name",
    type=str,
    help="Explainer factory function name (for custom explanation methods).",
    default="build_combinatorial_optimization_explainer",
)
@click.option("--output_path", type=click.Path(), default="explanations.json", help="Path to save explanations JSON (includes predictions, explanations, and timings).")
def explain(predictions_path, kg_name, kge_model_path, kge_config_path, lpx_config, output_path, factory_name):
    """Generate explanations using KELPIE, IMAGINE, CRIAGE, or other methods."""
    set_seeds(42)

    lpx_config = json.loads(lpx_config)
    factory = mhs_explain_factory if lpx_config["method"] == "mhs_explain" else build_combinatorial_optimization_explainer

    kge_config = read_json(kge_config_path)

    kg = KG(kg=kg_name, create_inverse_triples=kge_config["model"] == "ConvE")

    logger.info("Loading KGE model...")
    kge_model = load_kge_model(kge_model_path, kge_config, kg)
    kge_model.eval()
    kge_model.cuda()

    logger.info("Reading predictions")
    with open(predictions_path, "r", encoding="utf-8") as predictions:
        predictions = [x.strip().split("\t") for x in predictions.readlines()]

    explanations, times = run_explain(predictions, kg, kge_model, kge_config, lpx_config, factory)

    if lpx_config["method"] != "mhs_explain":
        predictions = [kg.label_triple(torch.tensor(kg.id_triple(p))) for p in predictions]
        explanations = [kg.label_triples(e) for e in explanations]
    output = [{"prediction": predictions[i], "explanation": explanations[i], "time": times[i]} for i in range(len(predictions))]
    write_json(output, output_path)
    logger.info(f"Explanation completed. Results saved to {output_path}")


@cli.command(
    help="Evaluate explanation quality using the LP-DIXIT protocol."
)
@click.option(
    "--explanations_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSON file with explanations to evaluate.",
)
@click.option("--kg_name", required=True, help=HELP_KG_NAME)
@click.option(
    "--kge_model_path", type=click.Path(exists=True), required=True, help=HELP_KGE_MODEL_PATH
)
@click.option(
    "--kge_config_path", type=click.Path(exists=True), required=True, help=HELP_KGE_CONFIG_PATH
)
@click.option("--output_path", type=click.Path(), default="evaluations.json", help=HELP_OUTPUT_PATH)
def evaluate(explanations_path, kg_name, kge_model_path, kge_config_path, output_path):
    """Evaluate explanations using LP-DIXIT protocol."""
    set_seeds(42)
    
    logger.info(f"Starting evaluation for {kg_name}")

    explanations = read_json(explanations_path)

    kge_config = read_json(kge_config_path)

    kg = KG(kg=kg_name, create_inverse_triples=kge_config["model"] == "ConvE")

    logger.info("Loading KGE model...")
    kge_model = load_kge_model(kge_model_path, kge_config, kg)
    kge_model.eval()
    kge_model.cuda()

    evaluations = run_evaluate(explanations, kg_name, True, kg, kge_model)

    write_json(evaluations, output_path)
    logger.info(f"Evaluation completed. Results saved to {output_path}")


@cli.command()
def comparison():
    """Run complete benchmarking workflow from comparison_setup.csv."""
    luigi.build([Comparison()], local_scheduler=True)


if __name__ == "__main__":
    cli()
