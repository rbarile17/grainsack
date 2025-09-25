"""This module implements the operation `explain`, the components involved
and the function factory for building the explain function from the explanation config."""

import operator
from ast import literal_eval
from functools import partial
from itertools import combinations

import time

import pandas as pd
import torch
from tqdm import tqdm

from . import (
    BISIMULATION,
    CRIAGE,
    DATA_POISONING,
    FR200K,
    FRUNI,
    GROUND_TRUTH,
    KELPIE,
    KGS_PATH,
    NECESSARY,
    SIMULATION,
    SUFFICIENT,
)
from .kge_lp import rank
from .relevance import (
    add_statements,
    criage_relevance,
    dp_relevance,
    estimate_rank_variation,
    remove_statements,
)
from .sift import criage_sift, topology_sift
from .summarize import bisimulation_summary, simulation
from .utils import load_kge_model


def run_explain(predictions, kg, kge_model_path, kge_config, lpx_config, build_explainer):
    """Compute the explanations for the given predictions based on the given data, models, and config.

    Compute the explanations for the given KG (predictions) using the statements in the other given KG,
    based on the given KGE model and the associated hyperparameter config (used for making the predictions),
    and according to the given explanation config.

    :param predictions: The predictions to be explained.
    :type predictions: list[tuple]
    :param kg: The KG for explaining the predictions.
    :type kg: KG
    :param kge_model_path: The path to the KGE .pt model file to be possibly used (depending on the explanation method) for explaining the predictions.
    :type kge_model_path: pathlib.Path
    :param kge_config_path: The path to the KGE .json config file to be possibly used (depending on the explanation method) for explaining the predictions.
    :type kge_config_path: pathlib.Path
    :param lpx_config: The explanation config dict. It should contain the method and the parameters for the explanation method.
    :type lpx_config: dict
    :return: The list of explanations each as a tensor of triples.
    :rtype: list
    """

    explain_partial = build_explainer(kg, kge_model_path, kge_config, lpx_config)

    explanations = []
    times = []
    for prediction in tqdm(predictions):
        start_time = time.perf_counter()
        explanation = explain_partial(prediction)
        end_time = time.perf_counter()
        explanations.extend(explanation)
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return explanations, times


def build_combinatorial_optimization_explainer(kg, kge_model_path, kge_config, lpx_config):
    """Function factory building the explanation method from the explanation config."""

    method = lpx_config["method"]
    if method in [KELPIE, DATA_POISONING, CRIAGE]:
        mode = lpx_config["mode"]
        summarization = lpx_config["summarization"]

        print(f"Loading KGE model...")
        kge_model = load_kge_model(kge_model_path, kge_config, kg)
        kge_model.eval()
        kge_model.cuda()

        if mode == NECESSARY:
            select_replication_entities_partial = lambda prediction: prediction[0].unsqueeze(0)
        elif mode == SUFFICIENT:
            select_replication_entities_partial = partial(select_replication_entities, kg, kge_model)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if summarization == SIMULATION:
            summarize_partial = partial(simulation, kg)
        elif summarization == BISIMULATION:
            summarize_partial = partial(bisimulation_summary, kg)
        else:
            summarize_partial = lambda x: (x, [])

        if method == KELPIE:
            max_length = 4
            sift_partial = partial(topology_sift, kg)
            fuse = remove_statements if mode == NECESSARY else add_statements
            relevance_partial = partial(estimate_rank_variation, kg, kge_model, kge_config, fuse, mode=mode)
        elif method == CRIAGE:
            sift_partial = partial(criage_sift, kg)
            relevance_partial = partial(criage_relevance, kg, kge_model, mode=mode)
            max_length = 1
        elif method == DATA_POISONING:
            sift_partial = partial(topology_sift, kg)
            aggregate = operator.add if mode == NECESSARY else operator.sub
            relevance_partial = partial(dp_relevance, kge_model, kge_config["optimizer_kwargs"]["lr"], aggregate)
            max_length = 1
        else:
            raise ValueError(f"Unknown method: {method}")

        explain_partial = partial(
            run_combinatorial_optimization,
            relevance_partial,
            sift_partial,
            summarize_partial,
            select_replication_entities_partial,
            max_length=max_length,
            kelpie=(method == KELPIE),
        )
    elif method == GROUND_TRUTH:
        if kg.name == FRUNI:
            explained_predictions = pd.read_csv(
                KGS_PATH / "FRUNI" / "explanations.txt",
                sep="\t",
                names=["prediction", "explanation", "fsv"],
                converters={"explanation": literal_eval, "prediction": literal_eval},
            )

            explain_partial = partial(get_explanation_from_fruni, explained_predictions, kg)
        elif kg.name == FR200K:
            explain_partial = partial(same_subject_baseline, kg)

            explained_predictions = pd.read_csv(
                KGS_PATH / "FR200K" / "explanations.txt",
                sep="\t",
                names=["s", "p", "o", "explanation", "fsv"],
                converters={"explanation": literal_eval},
            )
            explain_partial = partial(get_explanation_from_fr200k, explained_predictions, kg)
        else:
            raise ValueError(f"Unknown ground truth: {method}")
    else:
        raise ValueError(f"Unknown method: {method}")
    return explain_partial


def select_replication_entities(kg, kge_model, prediction, k=3):
    """Select the entities for replicating the prediction.

    Select the entities for replicating the prediction,
    thus obtaining the predictions to be used for the sufficient relevance
    (in combinatorial optimization LP-X methods).
    Specifically, select the entities in the KG that:
    - are different from the prediction subject;
    - do not result in a existing training triple or in a triple whose rank is equal to 1 if employed as prediction subject.
    """
    kge_model.eval()
    kge_model.cuda()

    entities = torch.arange(kg.num_entities).cuda()

    mask = entities == prediction[0]
    entities = entities[~mask]

    replications = prediction.unsqueeze(0).repeat(entities.size(0), 1)
    replications[:, 0] = entities
    replications[:, 1] = prediction[1]
    replications[:, 2] = prediction[2]
    training_triples = kg.training_triples.unsqueeze(0)[:, :1000, :]
    mask = (replications.unsqueeze(1) == training_triples).all(dim=-1).any(dim=-1)
    replications = replications[~mask]
    entities = entities[~mask]
    ranks = rank(kge_model, replications, [kg.training_triples, kg.validation_triples])
    entities = entities[ranks != 1]

    indices = torch.randperm(entities.size(0))[:k]
    entities = entities[indices]

    return entities


def run_combinatorial_optimization(
    relevance, get_statements, summarize, select_replication_entities_partial, prediction, max_length=2, kelpie=True
):
    """Explain the prediction via combinatorial optimization.

    Select the most relevant statements related to the prediction.

    :param relevance: function computing the relevance of each statement with respect to the prediction.
    :param get_statements: function to retrieve the statements related to the prediction.
    :param prediction: the prediction to be explained.
    :type prediction: torch.Tensor
    :param max_explanation_length: the maximum number of statements in the explanation.
    """
    prediction = torch.tensor(prediction).cuda()
    original_statements = get_statements(prediction)
    statements, partition = summarize(original_statements)
    statements = statements.unsqueeze(1)

    replication_entities = select_replication_entities_partial(prediction)

    combo_scores = []

    for length in range(1, min(statements.size(0), max_length) + 1):
        idx = torch.tensor(list(combinations(range(statements.size(0)), length)))
        combos = statements[idx].squeeze(2)
        if kelpie:
            relevances = relevance(replication_entities, prediction, combos, original_statements, partition)
        else:
            relevances = relevance(replication_entities, prediction, combos)

        combo_scores.extend(zip(combos, relevances))

    best_combo, _ = max(combo_scores, key=lambda x: x[1])

    best_combo = best_combo.reshape(-1, 3)
    mapped_statement = []
    for i, p, j in best_combo:
        if partition == []:
            mapped_statement.append((i, p, j))
        else:
            mapped_statement.extend([(s.item(), p.item(), o.item()) for s in partition[i] for o in partition[j]])

    mapped_statement = torch.tensor(mapped_statement, dtype=torch.int)

    return [mapped_statement]


def get_explanation_from_fr200k(explained_predictions, kg, prediction):
    """Explain the prediction by retrieving the explanation from FR200K."""
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
    """Explain the prediction by retrieving the explanation from FRUNI."""
    prediction = kg.label_triple(prediction)

    explanation = explained_predictions[explained_predictions["prediction"] == prediction]

    return explanation.to_dict(orient="records")


def random_baseline(kg, length):
    """Select a random triple from the KG as explanation."""
    indices = torch.randperm(kg.training_triples.size(0))[:length]
    return kg.training_triples[indices]


def same_subject_baseline(kg, prediction, length):
    """Explain the prediction by selecting a random triple with the same subject as the prediction."""
    training_triples = kg.training_triples[kg.training_triples[:, 0] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]


def same_predicate_baseline(kg, prediction, length):
    """Explain the prediction by selecting a random triple with the same predicate as the prediction."""
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]


def same_object_baseline(kg, prediction, length):
    """Explain the prediction by selecting a random triple with the same object as the prediction."""
    training_triples = kg.training_triples[kg.training_triples[:, 1] == prediction[0]]
    indices = torch.randperm(training_triples.size(0))[:length]
    return training_triples[indices]
