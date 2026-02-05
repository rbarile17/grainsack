"""Sift KG triples based on the prediction."""

import networkx as nx
import torch
from joblib import Parallel, delayed

from grainsack import DEVICE

from .kg import KG


def criage_sift(kg: KG, prediction, k: int = 20):
    """Extract top-k KG triples where the prediction's subject/object appears as object.

    Selects triples from the training set where either the subject or object of
    the prediction appears in the object position of the triple.

    Args:
        kg (KG): Knowledge graph containing training triples.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).
        k (int, optional): Maximum number of triples to return. Defaults to 20.

    Returns:
        torch.Tensor: Top-k filtered triples of shape (k, 3).
    """
    mask = (kg.training_triples[:, 2] == prediction[0]) | (
        kg.training_triples[:, 2] == prediction[2])
    return kg.training_triples[mask][:k]


def topology_sift(kg, prediction, triples, k: int = 10, n_jobs: int = -1):
    """Select top-k triples with highest fitness based on graph topology.

    Filters triples featuring the prediction's subject, ranking them by shortest
    path distance to the prediction's object (lower distance = higher fitness).

    Args:
        kg: Knowledge graph with NetworkX graph representation.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).
        triples (torch.Tensor): Candidate triples to filter.
        k (int, optional): Maximum number of triples to return. Defaults to 10.
        n_jobs (int, optional): Number of parallel jobs. -1 uses all cores. Defaults to -1.

    Returns:
        torch.Tensor: Top-k triples sorted by fitness, shape (k, 3).
    """

    def _compute_single_path(nx_graph, entity, target):
        try:
            return nx.shortest_path_length(nx_graph, entity, target)
        except nx.NetworkXNoPath:
            return 1e6

    if len(triples) == 0:
        return triples

    nx_graph = kg.nx_graph
    pred_subj = prediction[0].item()
    target = prediction[2].item()

    entities = [triple[2].item() if triple[0].item() == pred_subj else triple[0].item() for triple in triples]

    fitness_scores = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_compute_single_path)(nx_graph, entity, target) for entity in entities
    )

    sorted_indices = sorted(range(len(fitness_scores)),
                            key=lambda i: fitness_scores[i])[:k]
    result = torch.vstack([triples[i] for i in sorted_indices])

    return result


def get_statements(kg, prediction):
    """Extract all KG triples that feature the prediction's subject.

    Returns triples where the prediction's subject appears in either the
    subject or object position.

    Args:
        kg: Knowledge graph containing training triples.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).

    Returns:
        torch.Tensor: Filtered triples featuring the prediction's subject.
    """
    mask = (kg.training_triples[:, 0] == prediction[0]) | (
        kg.training_triples[:, 2] == prediction[0])
    return kg.training_triples[mask]


def pack_triples(triples, max_p, max_o):
    """Pack triples into a 1D tensor for efficient set operations.

    Args:
        triples: Tensor of shape (N, 3) containing triples
        max_p: Maximum predicate ID
        max_o: Maximum object ID

    Returns:
        1D tensor of packed triples
    """
    B_o = max_o + 1
    B_p = max_p + 1
    return triples[:, 0] * (B_p * B_o) + triples[:, 1] * B_o + triples[:, 2]


def hypothesis(kg, summary, partition, prediction):
    """Generate hypothesis triples from simulation summary and prediction.

    Constructs new candidate triples by expanding the summary graph around the
    entity in the prediction, using the partition mapping to generate concrete
    triples from the summarized representation. Filters out the prediction itself
    and any triples already in the training set.

    Args:
        kg: Knowledge graph with training triples.
        summary (torch.Tensor): Summarized graph representation.
        partition (list): Mapping from summary entities to concrete entities.
        prediction (torch.Tensor): Prediction triple to generate hypotheses for.

    Returns:
        torch.Tensor: Generated hypothesis triples on GPU, excluding existing triples.
    """
    e = prediction[0]

    qe = next(i for i, t in enumerate(partition) if (t == e).any())

    rows_sub = summary[summary[:, 0] == qe]
    rows_obj = summary[summary[:, 2] == qe]

    blocks = []

    if len(rows_sub) > 0:
        for i in range(len(rows_sub)):
            p = rows_sub[i, 1]
            qo = rows_sub[i, 2].item()
            o = partition[qo]
            n = o.numel()
            if n == 0:
                continue
            block = torch.empty((n, 3), dtype=prediction.dtype, device=DEVICE)
            block[:, 0] = e
            block[:, 1] = p
            block[:, 2] = o
            blocks.append(block)

    if len(rows_obj) > 0:
        for i in range(len(rows_obj)):
            qs = rows_obj[i, 0].item()
            p = rows_obj[i, 1]
            s = partition[qs]
            n = s.numel()
            if n == 0:
                continue
            block = torch.empty((n, 3), dtype=prediction.dtype, device=DEVICE)
            block[:, 0] = s
            block[:, 1] = p
            block[:, 2] = e
            blocks.append(block)

    triples = torch.cat(blocks, dim=0)

    triples = triples[(triples != prediction).any(dim=1)]

    train = kg.training_triples
    max_p = max(triples[:, 1].max(), train[:, 1].max()).item()
    max_o = max(triples[:, 2].max(), train[:, 2].max()).item()

    packed_triples = pack_triples(triples, max_p, max_o)
    packed_train = pack_triples(train, max_p, max_o)

    mask_in_train = torch.isin(packed_triples, packed_train)
    triples = triples[~mask_in_train].to(triples.dtype)

    return triples
