"""Sift KG triples based on the prediction."""

import networkx as nx
import torch

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


def fitness(nx_kg, prediction, triple):
    """Compute the fitness score of a triple based on shortest path distance.

    For a prediction <s, p, o> and triple <s, q, e> (or <e, q, s>), fitness is
    defined as the length of the shortest path between entity e and the
    prediction's object o. Lower values indicate higher fitness.

    Args:
        nx_kg (networkx.MultiGraph): Knowledge graph as a NetworkX graph.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).
        triple (torch.Tensor): Triple to compute fitness for.

    Returns:
        float: Shortest path length, or 1e6 if no path exists.
    """
    entity = triple[2] if triple[0] == prediction[0] else triple[0]
    entity = entity.item()

    try:
        return nx.shortest_path_length(nx_kg, entity, prediction[2].item())
    except nx.NetworkXNoPath:
        return 1e6


def topology_sift(kg, prediction, triples, k: int = 10):
    """Select top-k triples with highest fitness based on graph topology.

    Filters triples featuring the prediction's subject, ranking them by shortest
    path distance to the prediction's object (lower distance = higher fitness).

    Args:
        kg: Knowledge graph with NetworkX graph representation.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).
        triples (torch.Tensor): Candidate triples to filter.
        k (int, optional): Maximum number of triples to return. Defaults to 10.

    Returns:
        torch.Tensor: Top-k triples sorted by fitness, shape (k, 3).
    """
    return torch.vstack(sorted(triples, key=lambda x: fitness(kg.nx_graph, prediction, x))[:k])


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
    partition = [p for p in partition]

    qe = next(i for i, t in enumerate(partition) if (t == e).any())

    rows_sub = summary[summary[:, 0] == qe]
    rows_obj = summary[summary[:, 2] == qe]

    blocks = []

    # Build blocks for subject position
    for _, p, qo in rows_sub.tolist():
        o = partition[qo]
        n = o.numel()
        if n == 0:
            continue
        block = torch.empty((n, 3), dtype=prediction.dtype).cpu()
        block[:, 0] = e
        block[:, 1] = p
        block[:, 2] = o
        blocks.append(block)

    # Build blocks for object position
    for qs, p, _ in rows_obj.tolist():
        s = partition[qs]
        n = s.numel()
        if n == 0:
            continue
        block = torch.empty((n, 3), dtype=prediction.dtype).cpu()
        block[:, 0] = s
        block[:, 1] = p
        block[:, 2] = e
        blocks.append(block)

    triples = torch.cat(blocks, dim=0)

    # Filter out the prediction itself
    triples = triples[(triples != prediction).any(dim=1)]

    # Filter out existing training triples
    train = kg.training_triples.to(triples.device)
    max_p = max(triples[:, 1].max(), train[:, 1].max()).item()
    max_o = max(triples[:, 2].max(), train[:, 2].max()).item()

    packed_triples = pack_triples(triples, max_p, max_o)
    packed_train = pack_triples(train, max_p, max_o)

    mask_in_train = torch.isin(packed_triples, packed_train)
    triples = triples[~mask_in_train].to(triples.dtype)

    return triples.to(DEVICE)
