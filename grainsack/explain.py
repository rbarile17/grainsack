"""This module contains implementation of explanation methods compoenntes, function factory an orchestrator."""

import math
import operator
import random

from ast import literal_eval
from functools import partial
from itertools import combinations

import pandas as pd
import torch
from tqdm import tqdm

from . import (
    EXPLANATIONS_PATH,
    KGS_PATH,
    LP_CONFIGS_PATH,
    PREDICTIONS_PATH,
    KELPIE,
    DATA_POISONING,
    CRIAGE,
    NECESSARY,
    SUFFICIENT,
    SIMULATION,
    BISIMULATION,
    FRUNI,
    FR200K,
    GROUND_TRUTH
)
from .kg import KG
from .lp import evaluate
from .relevance import (
    add_statements,
    criage_relevance,
    dp_relevance,
    estimate_rank_variation,
    remove_statements,
)
from .sift import criage_sift, topology_sift
from .summarize import simulation, bisimulation
from .utils import load_model, read_json, write_json


def run_explanation_task(model, kg, method, explanation_kwargs, build_explainer):
    """Explanation task orchestrator"""
    kwargs_str = list(explanation_kwargs.values())
    kwargs_str = [x if x is not None else "null" for x in kwargs_str]
    kwargs_str = "_".join(kwargs_str)
    output_file = f"{model}_{kg}_{method}_{kwargs_str}.json"

    print("Loading predictions...")
    predictions = PREDICTIONS_PATH / f"{model}_{kg}2.csv"
    with open(predictions, "r") as predictions:
        predictions = [x.strip().split("\t") for x in predictions.readlines()]

    print(f"Loading KG {kg}...")
    kg = KG(kg=kg)

    predictions = kg.id_triples(predictions)

    explain = build_explainer(model, kg, method, explanation_kwargs)

    explanations = []

    for prediction in tqdm(predictions):
        explanation = explain(prediction)
        explanations.extend(explanation)
    write_json(explanations, EXPLANATIONS_PATH / output_file)


def build_combinatorial_optimization_explainer(model, kg, method, explanation_kwargs):
    """Function factory"""
    if method in [KELPIE, DATA_POISONING, CRIAGE]:
        mode = explanation_kwargs["mode"]
        summarization = explanation_kwargs["summarization"]
        lp_config_path = LP_CONFIGS_PATH / f"{model}_{kg.name}.json"
        lp_config = read_json(lp_config_path)
        # lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]

        print(f"Loading model {model}...")
        model = load_model(lp_config, kg)
        model.eval()
        model.cuda()

        if mode == NECESSARY:
            select_replication_entities_partial = lambda prediction: prediction[0].unsqueeze(0)
        elif mode == SUFFICIENT:
            select_replication_entities_partial = partial(select_replication_entities, kg, model)
        if summarization == SIMULATION:
            summarize_partial = partial(simulation, kg)
        elif summarization == BISIMULATION:
            summarize_partial = partial(bisimulation, kg)
        else:
            summarize_partial = lambda x: (x, [])

        if method == KELPIE:
            sift_partial = partial(topology_sift, kg)
            if mode == NECESSARY:
                fuse = remove_statements
            elif mode == SUFFICIENT:
                fuse = add_statements

            relevance_partial = partial(estimate_rank_variation, kg, model, lp_config["model_kwargs"], fuse)
        if method == CRIAGE:
            sift_partial = partial(criage_sift, kg)
            relevance_partial = partial(criage_relevance, kg, model)
            summarize_partial = lambda x, y: y
        if method == DATA_POISONING:
            aggregate = operator.add if mode == NECESSARY else operator.sub
            relevance_partial = partial(dp_relevance, model, lp_config["optimizer_kwargs"]["lr"], aggregate)

        explain_partial = partial(
            run_combinatorial_optimization,
            relevance_partial,
            sift_partial,
            summarize_partial,
            select_replication_entities_partial,
        )
    elif method == GROUND_TRUTH:
        if kg.name == FRUNI:
            explained_predictions = pd.read_csv(
                KGS_PATH / "FRUNI" / "data.txt",
                sep="\t",
                names=["prediction", "explanation", "fsv"],
                converters={"explanation": literal_eval, "prediction": literal_eval},
            )

            explain_partial = partial(get_explanation_from_fruni, explained_predictions, kg)
        elif kg.name == FR200K:
            explain_partial = partial(same_subject_baseline, kg)

            explained_predictions = pd.read_csv(
                KGS_PATH / "FR200K" / "data.txt",
                sep="\t",
                names=["s", "p", "o", "explanation", "fsv"],
                converters={"explanation": literal_eval},
            )
            explain_partial = partial(get_explanation_from_fr200k, explained_predictions, kg)

    return explain_partial


def select_replication_entities(kg, model, prediction, k=3):
    """Select the entities that are different from the prediction subject;
    that if employed as prediction subject do not result in a training triple or in a triple predicted by the model.

    The replication entities.
    """
    model.eval()
    model.cuda()

    entities = torch.arange(kg.num_entities).cuda()

    mask = entities == prediction[0]
    entities = entities[~mask]

    replications = prediction.unsqueeze(0).repeat(entities.size(0), 1)
    replications[:, 0] = entities
    replications[:, 1] = prediction[1]
    replications[:, 2] = prediction[2]
    training_triples = kg.training_triples.unsqueeze(0)
    mask = (replications.unsqueeze(1) == training_triples).all(dim=-1).any(dim=-1)
    replications = replications[~mask]
    entities = entities[~mask]
    replications = replications[:10]
    ranks = evaluate(model, replications, [kg.training_triples, kg.validation_triples])
    # entities = entities[ranks != 1]

    indices = torch.randperm(entities.size(0))[:k]
    entities = entities[indices]

    return entities


def run_combinatorial_optimization(relevance, get_statements, summarize, select_replication_entities, prediction, max_length=4):
    """Explain the prediction via combinatorial optimization.

    Select the most relevant statement related to the prediction.

    Args:
        relevance: function computing the relevance of each statement with respect to the prediction.
        get_statements: function to retrieve the statements related to the prediction.
        prediction: the prediction to be explained.
        max_explanation_length: the maximum number of statements in the explanation.

    Returns:
        The explanation.
    """
    prediction = torch.tensor(prediction).cuda()
    original_statements = get_statements(prediction)
    statements, partition = summarize(original_statements)
    statements = statements.unsqueeze(1)

    replication_entities = select_replication_entities(prediction)

    relevances = relevance(replication_entities, prediction, statements, original_statements, partition)

    for length in range(4, min(statements.size(0), max_length) + 1):
        idx = torch.tensor(list(combinations(range(statements.size(0)), length)))
        combos = statements[idx].squeeze(2)
        cur_relevances = relevance(replication_entities, prediction, combos, original_statements, partition)

    return []


def get_explanation_from_fr200k(explained_predictions, kg, prediction):
    """"Retrieve the explanation for a prediction from the FR200K dataset ground-truth."""
    s, p, o = prediction
    s = kg.id_to_entity[s]
    p = kg.id_to_relation[p]
    o = kg.id_to_entity[o]

    explanation = explained_predictions[
        (explained_predictions["s"] == s) & (explained_predictions["p"] == p) & (explained_predictions["o"] == o)
    ]

    explanation["prediction"] = explanation.apply(lambda row: (row["s"], row["p"], row["o"]), axis=1)
    explanation.drop(columns=["s", "p", "o"], inplace=True)

    return explanation.to_dict(orient="records")


def get_explanation_from_fruni(explained_predictions, kg, prediction):
    """"Retrieve the explanation for a prediction from the FRUNI dataset ground-truth."""
    prediction = kg.label_triple(prediction)

    explanation = explained_predictions[explained_predictions["prediction"] == prediction]

    return explanation.to_dict(orient="records")


def random_baseline(kg, length):
    """"Random baseline"""
    indices = torch.randperm(kg.training_triples.size(0))[:length]
    return kg.training_triples[indices]


def same_subject_baseline(kg, prediction, length):
    """"Random baseline selecting the same subject as the prediction"""
    training_triples = kg.training_triples[kg.training_triples[:, 0] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]


def same_predicate_baseline(kg, prediction, length):
    """"Random baseline selecting the same predicate as the prediction"""
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]


def same_object_baseline(kg, prediction):
    """"Random baseline selecting the same object as the prediction"""
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]
