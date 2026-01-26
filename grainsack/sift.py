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


def topology_sift(kg, prediction, triples, k: int = 5):
    """Get the KG triples featuring the subject of the prediction and having the highest fitness."""
    return torch.vstack(sorted(triples, key=lambda x: fitness(kg.nx_graph, prediction, x))[:k])


def get_statements(kg, prediction):
    mask = (kg.training_triples[:, 0] == prediction[0]) | (kg.training_triples[:, 2] == prediction[0])
    triples = kg.training_triples[mask]
    return triples


def hypothesis(kg, summary, partition, prediction):
    e = prediction[0].cpu()
    summary = summary.cpu()
    partition = [p.cpu() for p in partition]

    qe = next(i for i, t in enumerate(partition) if (t == e).any())

    rows_sub = summary[summary[:, 0] == qe]
    rows_obj = summary[summary[:, 2] == qe]

    blocks = []

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

    triples = triples[(triples != prediction.cpu()).any(dim=1)]

    train = kg.training_triples.to(triples.device)

    max_p = max(triples[:,1].max(), train[:,1].max()).item()
    max_o = max(triples[:,2].max(), train[:,2].max()).item()

    B_o = max_o + 1
    B_p = max_p + 1

    def pack(t):
        return t[:,0] * (B_p * B_o) + t[:,1] * B_o + t[:,2]

    packed_triples = pack(triples)
    packed_train   = pack(train)

    mask_in_train = torch.isin(packed_triples, packed_train)
    triples = triples[~mask_in_train].to(triples.dtype)

    # is_training = (triples[:, None, :] == train[None, :, :]).all(dim=2).any(dim=1)
    # triples = triples[~is_training]

    return triples.cuda()
