"""This module implements the operation `explain`, the components involved
and the function factory for building the explain function from the explanation config."""

from functools import partial
from itertools import combinations

import time

import torch
from tqdm import tqdm

from . import BISIMULATION, CRIAGE, DATA_POISONING, KELPIE, SIMULATION, IMAGINE
from .relevance import criage_relevance, dp_relevance, estimate_rank_variation
from .sift import criage_sift, topology_sift, get_statements, hypothesis
from .summarize import bisimulation_summary, simulation_summary


def run_explain(predictions, kg, kge_model, kge_config, lpx_config, build_explainer):
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
    explain_partial = build_explainer(kg, kge_model, kge_config, lpx_config)

    explanations = []
    times = []
    for i, prediction in enumerate(predictions):
        print(f"Explaining prediction {i+1}/{len(predictions)}")
        start_time = time.perf_counter()
        explanation = explain_partial(prediction)
        end_time = time.perf_counter()
        explanations.extend(explanation)
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    return explanations, times


def build_combinatorial_optimization_explainer(kg, kge_model, kge_config, lpx_config):
    """Function factory building the explanation method from the explanation config."""

    method = lpx_config["method"]
    summarization = lpx_config["summarization"]

    operation = None

    if method == IMAGINE:
        max_length = 2
        print("Computing simulation summary...", time.strftime("%H:%M:%S"))
        summary, partition = simulation_summary(kg, kg.training_triples)

        print("End simulation summary", time.strftime("%H:%M:%S"))

        get_statements_partial = partial(hypothesis, kg, summary, partition)
        sift_partial = partial(topology_sift, kg)
        relevance_partial = partial(estimate_rank_variation, kg, kge_model, kge_config)

        operation = "add"

        if summarization == SIMULATION:
            summarize_partial = partial(simulation_summary, kg)
        elif summarization == BISIMULATION:
            summarize_partial = partial(bisimulation_summary, kg)
        else:
            summarize_partial = lambda x: (x, [])
    elif method == KELPIE:
        operation = "remove"
        max_length = 2
        if summarization == SIMULATION:
            summarize_partial = partial(simulation_summary, kg)
        elif summarization == BISIMULATION:
            summarize_partial = partial(bisimulation_summary, kg)
        else:
            summarize_partial = lambda x: (x, [])

        get_statements_partial = partial(get_statements, kg)
        sift_partial = partial(topology_sift, kg)
        relevance_partial = partial(estimate_rank_variation, kg, kge_model, kge_config)
    elif method == CRIAGE:
        summarize_partial = lambda x: (x, [])
        get_statements_partial = partial(criage_sift, kg)
        sift_partial = lambda x, y: y
        relevance_partial = partial(criage_relevance, kg, kge_model)
        max_length = 1
    elif method == DATA_POISONING:
        summarize_partial = lambda x: (x, [])
        get_statements_partial = partial(get_statements, kg)
        sift_partial = partial(topology_sift, kg)
        relevance_partial = partial(dp_relevance, kge_model, kge_config["optimizer_kwargs"]["lr"])
        max_length = 1
    else:
        raise ValueError(f"Unknown method: {method}")

    explain_partial = partial(
        run_combinatorial_optimization,
        kg,
        relevance_partial,
        get_statements_partial,
        sift_partial,
        summarize_partial,
        max_length=max_length,
        kelpie=(method in [KELPIE, IMAGINE]),
        operation=operation,
    )
    return explain_partial


def run_combinatorial_optimization(
    kg, relevance, get_statements, sift, summarize, prediction, max_length=2, kelpie=True, operation=None
):
    """Explain the prediction via combinatorial optimization.

    Select the most relevant statements related to the prediction.

    :param relevance: function computing the relevance of each statement with respect to the prediction.
    :param get_statements: function to retrieve the statements related to the prediction.
    :param prediction: the prediction to be explained.
    :type prediction: torch.Tensor
    :param max_explanation_length: the maximum number of statements in the explanation.
    """
    try:
        print("Explaining prediction:", prediction)
        prediction = kg.id_triple(prediction)
        prediction = torch.tensor(prediction).cuda()

        print("Getting statements...", time.strftime("%H:%M:%S"))

        original_statements = get_statements(prediction)
        original_statements = sift(prediction, original_statements)

        print("Summarizing statements...", time.strftime("%H:%M:%S"))
        statements, partition = summarize(original_statements)
        statements = statements.unsqueeze(1)

        if statements.size(0) == 0:
            print("No statements found.")
            return [[]]

        for length in range(1, min(statements.size(0), max_length) + 1):
            print(f"Evaluating combinations of length {length}...", time.strftime("%H:%M:%S"))
            idx = torch.tensor(list(combinations(range(statements.size(0)), length)))
            combos = statements[idx].squeeze(2)
            if kelpie:
                relevances = relevance(prediction, combos, original_statements, partition, operation=operation)
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
    except Exception as e:
        print(f"An error occurred during explanation: {e}")
        return [[]]