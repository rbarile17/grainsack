"""This module implements the operation `explain`, the components involved
and the function factory for building the explain function from the explanation config."""

import time
from functools import partial
from itertools import combinations

import torch

from . import CRIAGE, DATA_POISONING, IMAGINE, KELPIE, SIMULATION
from .relevance import criage_relevance, dp_relevance, estimate_rank_variation
from .sift import criage_sift, get_statements, hypothesis, topology_sift
from .summarize import simulation_summary

NO_SUMMARIZE = lambda x: (x, [])


def run_explain(predictions, kg, kge_model, kge_config, lpx_config, build_explainer):
    """Generate explanations for multiple predictions.
    
    Computes explanations for given predictions using the specified KG, KGE model,
    and explanation method configured through the factory function.
    
    Args:
        predictions (list): List of prediction triples to explain.
        kg (KG): Knowledge graph for generating explanations.
        kge_model: Trained KGE model for relevance computation.
        kge_config (dict): Configuration dictionary for the KGE model.
        lpx_config (dict): Explanation method configuration containing 'method'
            and method-specific parameters.
        build_explainer (callable): Factory function that builds the explanation
            function from kg, kge_model, kge_config, and lpx_config.
            
    Returns:
        tuple: (explanations, times) where:
            - explanations (list): List of explanation tensors, one per prediction.
            - times (list): Computation time in seconds for each explanation.
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
    """Factory function that builds an explanation method from configuration.
    
    Creates a configured explanation function based on the specified method
    (KELPIE, IMAGINE, CRIAGE, or DATA_POISONING). Each method combines different
    components for statement extraction, filtering, summarization, and relevance
    computation.
    
    Args:
        kg (KG): Knowledge graph for explanation generation.
        kge_model: Trained KGE model for relevance computation.
        kge_config (dict): KGE model configuration including hyperparameters.
        lpx_config (dict): Explanation configuration with keys:
            - 'method' (str): One of KELPIE, IMAGINE, CRIAGE, DATA_POISONING.
            - 'summarization' (str): Summarization strategy (e.g., SIMULATION).
            
    Returns:
        callable: Configured explanation function that takes a prediction and
            returns explanation triples.
            
    Raises:
        ValueError: If the specified method is not supported.
    """

    method = lpx_config["method"]
    summarization = lpx_config["summarization"]

    def get_summarize_fn():
        return partial(simulation_summary, kg) if summarization == SIMULATION else NO_SUMMARIZE

    configs = {
        IMAGINE: {
            "max_length": 2,
            "operation": "add",
            "kelpie": True,
            "setup": simulation_summary(kg, kg.training_triples),
        },
        KELPIE: {
            "max_length": 2,
            "operation": "remove",
            "kelpie": True,
            "get_statements": partial(get_statements, kg),
            "sift": partial(topology_sift, kg),
            "relevance": partial(estimate_rank_variation, kg, kge_model, kge_config),
        },
        CRIAGE: {
            "max_length": 1,
            "operation": None,
            "kelpie": False,
            "get_statements": partial(criage_sift, kg),
            "sift": lambda x, y: y,
            "relevance": partial(criage_relevance, kg, kge_model),
            "summarize": NO_SUMMARIZE,
        },
        DATA_POISONING: {
            "max_length": 1,
            "operation": None,
            "kelpie": False,
            "get_statements": partial(get_statements, kg),
            "sift": partial(topology_sift, kg),
            "relevance": partial(dp_relevance, kge_model, kge_config["optimizer_kwargs"]["lr"]),
            "summarize": NO_SUMMARIZE,
        },
    }

    if method not in configs:
        raise ValueError(f"Unknown method: {method}")

    config = configs[method]

    if method == IMAGINE:
        summary, partition = config["setup"]
        get_statements_partial = partial(hypothesis, kg, summary, partition)
        sift_partial = partial(topology_sift, kg)
        relevance_partial = partial(estimate_rank_variation, kg, kge_model, kge_config)
        summarize_partial = get_summarize_fn()
    else:
        get_statements_partial = config["get_statements"]
        sift_partial = config["sift"]
        relevance_partial = config["relevance"]
        summarize_partial = config.get("summarize", get_summarize_fn())

    return partial(
        run_combinatorial_optimization,
        kg,
        relevance_partial,
        get_statements_partial,
        sift_partial,
        summarize_partial,
        max_length=config["max_length"],
        kelpie=config["kelpie"],
        operation=config["operation"],
    )


def run_combinatorial_optimization(
    kg, relevance, get_statements, sift, summarize, prediction, max_length=2, kelpie=True, operation=None
):
    """Explain a prediction via combinatorial optimization over candidate statements.
    
    Generates an explanation by: (1) extracting candidate statements related to
    the prediction, (2) filtering/sifting for relevant statements, (3) optionally
    summarizing, (4) evaluating combinations up to max_length, and (5) selecting
    the combination with highest relevance score.
    
    Args:
        kg (KG): Knowledge graph.
        relevance (callable): Function computing relevance scores for statement combinations.
        get_statements (callable): Function extracting candidate statements for a prediction.
        sift (callable): Function filtering statements based on topology/relevance.
        summarize (callable): Function summarizing statements, returns (statements, partition).
        prediction: Prediction triple to explain.
        max_length (int, optional): Maximum number of statements in explanation. Defaults to 2.
        kelpie (bool, optional): Whether to use Kelpie-style relevance computation. Defaults to True.
        operation (str, optional): Operation for relevance ('add' or 'remove'). Defaults to None.
        
    Returns:
        list: Single-element list containing the best explanation as a tensor of triples,
            or [[]] if no explanation is found or an error occurs.
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