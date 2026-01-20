"""Sift KG triples based on the prediction."""

import networkx as nx
import torch

from .kg import KG


def criage_sift(kg: KG, prediction, k: int = 20):
    """Get k KG triples featuring as object either the subject or object of the prediction."""
    mask = (kg.training_triples[:, 2] == prediction[0]) | (kg.training_triples[:, 2] == prediction[2])
    return kg.training_triples[mask][:k]


def fitness(nx_kg, prediction, triple):
    """Measure the length of the shortest path between the subject/object of the triple and the object of the prediction.

    Let <s, p, o> be the prediction, the fitness of a triple <s, q, e> (<e, q, s>)
      is the length of the shorthest path between e and o.
    """
    entity = triple[2] if triple[0] == prediction[0] else triple[0]
    entity = entity.item()

    try:
        return nx.shortest_path_length(nx_kg, entity, prediction[2].item())
    except nx.NetworkXNoPath:
        return 1e6


def topology_sift(kg, prediction, triples, k: int = 20):
    """Get the KG triples featuring the subject of the prediction and having the highest fitness."""
    return torch.vstack(sorted(triples, key=lambda x: fitness(kg.nx_graph, prediction, x))[:k])


def get_statements(kg, prediction):
    mask = (kg.training_triples[:, 0] == prediction[0]) | (kg.training_triples[:, 2] == prediction[0])
    triples = kg.training_triples[mask]
    return triples


def hypothesis(kg, summary, partition, prediction):
    qe = next(i for i, t in enumerate(partition) if (t == prediction[0]).any())

    qtriples = summary[(summary[:, 0] == qe) | (summary[:, 2] == qe)]
    qtriples = [(partition[qs], p, partition[qo]) for qs, p, qo in qtriples]

    triples = []
    for qs, p, qo in qtriples:
        if prediction[0] in qs:
            triples.extend([torch.tensor((prediction[0], p, o)).cuda() for o in qo])
        elif prediction[0] in qo:
            triples.extend([torch.tensor((s, p, prediction[0])).cuda() for s in qs])
    triples = [t for t in triples if (prediction != t).any()]
    triples = [t for t in triples if not (kg.training_triples == t).all(dim=1).any()]

    triples = torch.stack(triples, dim=0).cuda()

    return triples
