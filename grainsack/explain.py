"""This module implements the operation `explain`, the components involved
and the function factory for building the explain function from the explanation config."""

from ast import literal_eval
from functools import partial
from itertools import combinations

import time

import pandas as pd
import torch
from tqdm import tqdm

from . import BISIMULATION, CRIAGE, DATA_POISONING, FR200K, FRUNI, GROUND_TRUTH, KELPIE, KGS_PATH, SIMULATION
from .relevance import criage_relevance, dp_relevance, estimate_rank_variation
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
        summarization = lpx_config["summarization"]

        print(f"Loading KGE model...")
        kge_model = load_kge_model(kge_model_path, kge_config, kg)
        kge_model.eval()
        kge_model.cuda()

        if summarization == SIMULATION:
            summarize_partial = partial(simulation, kg)
        elif summarization == BISIMULATION:
            summarize_partial = partial(bisimulation_summary, kg)
        else:
            summarize_partial = lambda x: (x, [])

        if method == KELPIE:
            max_length = 2
            sift_partial = partial(topology_sift, kg)
            relevance_partial = partial(estimate_rank_variation, kg, kge_model, kge_config)
        elif method == CRIAGE:
            sift_partial = partial(criage_sift, kg)
            relevance_partial = partial(criage_relevance, kg, kge_model)
            max_length = 1
        elif method == DATA_POISONING:
            sift_partial = partial(topology_sift, kg)
            relevance_partial = partial(dp_relevance, kge_model, kge_config["optimizer_kwargs"]["lr"])
            max_length = 1
        else:
            raise ValueError(f"Unknown method: {method}")

        explain_partial = partial(
            run_combinatorial_optimization,
            relevance_partial,
            sift_partial,
            summarize_partial,
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


def run_combinatorial_optimization(relevance, get_statements, summarize, prediction, max_length=2, kelpie=True):
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

    for length in range(1, min(statements.size(0), max_length) + 1):
        idx = torch.tensor(list(combinations(range(statements.size(0)), length)))
        combos = statements[idx].squeeze(2)
        if kelpie:
            relevances = relevance(prediction, combos, original_statements, partition)
        else:
            relevances = relevance(prediction, combos)

    best_combo = combos[torch.argmax(relevances)]

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
