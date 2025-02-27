""""""

import math
import operator
import random
from ast import literal_eval
from functools import partial
from itertools import combinations

import pandas as pd
import torch
from tqdm import tqdm

from . import EXPLANATIONS_PATH, KGS_PATH, LP_CONFIGS_PATH, PREDICTIONS_PATH
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
from .summarize import simulation
from .utils import load_model, read_json, write_json


def run_explanation_task(kg: str, model: str, method: str, mode: str) -> None:
    lp_config_path = LP_CONFIGS_PATH / f"{model}_{kg}.json"
    lp_config = read_json(lp_config_path)
    # lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]

    output_file = f"{method}_{mode}_{model}_{kg}.json"

    print("Loading predictions...")
    predictions = PREDICTIONS_PATH / f"{model}_{kg}2.csv"
    with open(predictions, "r") as predictions:
        predictions = [x.strip().split("\t") for x in predictions.readlines()]

    print(f"Loading KG {kg}...")
    kg = KG(kg=kg)

    print(f"Loading model {model}...")
    model = load_model(lp_config, kg)
    model.eval()

    predictions = kg.id_triples(predictions)

    explanations = []

    explain = build_pipeline(model, kg, lp_config, method, mode)

    for prediction in tqdm(predictions):
        explanation = explain(prediction)
        explanations.extend(explanation)
    write_json(explanations, EXPLANATIONS_PATH / output_file)


def build_pipeline(model, kg: KG, lp_config, method: str, mode: str):
    """"""
    # sift_partial = partial(topology_sift, kg)
    # sift_partial = partial(criage_sift, kg)

    # fuse = add_statements
    # select_replication_entities_partial = lambda prediction: prediction[0].unsqueeze(0)
    # select_replication_entities_partial = partial(select_replication_entities, kg, model)

    # relevance_partial = partial(estimate_rank_variation, kg, model, lp_config["model_kwargs"], fuse)
    # relevance_partial = partial(criage_relevance, kg, model)
    # aggregate = operator.add
    # relevance_partial = partial(dp_relevance, model, lp_config["optimizer_kwargs"]["lr"], aggregate)

    # summarize_partial = partial(simulation, kg)

    # explain_partial = partial(run_combinatorial_optimization, relevance_partial, sift_partial, summarize_partial)

    # explain_partial = partial(same_subject_baseline, kg)

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

    The replication entities
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
    statements = original_statements
    statements, partition = summarize(prediction[0], original_statements)
    statements = statements.unsqueeze(1)

    replication_entities = select_replication_entities(prediction)

    relevances = relevance(replication_entities, prediction, statements, original_statements, partition)

    for length in range(4, min(statements.size(0), max_length) + 1):
        idx = torch.tensor(list(combinations(range(statements.size(0)), length)))
        combos = statements[idx].squeeze(2)
        cur_relevances = relevance(replication_entities, prediction, combos, original_statements, partition)


def get_explanation_from_fr200k(explained_triples, kg, prediction):
    s, p, o = prediction
    s = kg.id_to_entity[s]
    p = kg.id_to_relation[p]
    o = kg.id_to_entity[o]

    explanation = explained_triples[(explained_triples["s"] == s) & (explained_triples["p"] == p) & (explained_triples["o"] == o)]

    explanation["prediction"] = explanation.apply(lambda row: (row["s"], row["p"], row["o"]), axis=1)
    explanation.drop(columns=["s", "p", "o"], inplace=True)

    return explanation.to_dict(orient="records")


def random_baseline(kg, prediction):
    return kg.training_triples[math.ceil(random.uniform(0, kg.num_entities))]


def same_subject_baseline(kg, prediction):
    training_triples = kg.training_triples[kg.training_triples[:, 0] == prediction[0]]
    return training_triples[math.ceil(random.uniform(0, kg.num_entities))]


def same_predicate_baseline(kg, prediction):
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    return training_triples[math.ceil(random.uniform(0, kg.num_entities))]


def same_object_baseline(kg, prediction):
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    return training_triples[math.ceil(random.uniform(0, kg.num_entities))]
