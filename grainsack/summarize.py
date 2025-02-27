"""Compute the simulation and bisimulation of a knowledge graph."""

import networkx as nx
import torch
from bispy import compute_maximum_bisimulation


def simulation(kg, triples):
    """Compute the simulation of a (sub-)graph"""
    partition = kg.get_partition(triples)

    n_classes = len(partition)
    subjects = triples[:, 0]
    objects = triples[:, 2]

    subject_masks = [torch.isin(subjects, cl) for cl in partition]
    object_masks = [torch.isin(objects, cl) for cl in partition]
    pairs = [(i, j) for i in range(n_classes) for j in range(n_classes)]

    quotient_triples = []
    for i, j in pairs:
        predicates = torch.unique(triples[subject_masks[i] & object_masks[j]][:, 1])
        quotient_triples.extend([(i, p, j) for p in predicates])

    quotient_triples = torch.tensor(quotient_triples, dtype=torch.long)

    return quotient_triples, partition


def preprocess(multigraph):
    """Preprocess a multigraph to a directed graph."""
    digraph = nx.DiGraph()

    for s, o, data in multigraph.edges(data=True):
        p = data["label"]

        if not digraph.has_node(s):
            digraph.add_node(s)
        if not digraph.has_node(o):
            digraph.add_node(o)
        po_node = (p, o)
        if not digraph.has_node(po_node):
            digraph.add_node(po_node)

        digraph.add_edge(s, po_node)

    return digraph


def bisimulation_summary(kg, triples):
    """Compute the bisimulation of a (sub-)graph."""
    graph = nx.MultiDiGraph([(s.item(), o.item(), {"label": p.item()}) for s, p, o in triples])

    graph = preprocess(graph)
    partition = kg.get_partition(triples)
    partition = [frozenset(cl.tolist()) for cl in partition]

    for node in graph.nodes():
        if isinstance(node, tuple):
            partition.append(frozenset({node}))

    bisimulation = compute_maximum_bisimulation(graph, partition)
    bisimulation = [frozenset(cl) for cl in bisimulation]

    bisimulation = [cl for cl in bisimulation if not any(isinstance(node, tuple) for node in cl)]
    bisimulation_ = [list(cl) for cl in bisimulation]
    bisimulation = [torch.tensor(cl).cuda() for cl in bisimulation_]

    subject_masks = [torch.isin(triples[:, 0], cl) for cl in bisimulation]
    object_masks = [torch.isin(triples[:, 2], cl) for cl in bisimulation]
    pairs = [(i, j) for i in range(len(bisimulation)) for j in range(len(bisimulation))]

    quotient_triples = []
    for i, j in pairs:
        predicates = torch.unique(triples[subject_masks[i] & object_masks[j]][:, 1])
        filtered_predicates = [
            p
            for p in predicates
            if all(
                any(torch.isin(torch.tensor((s, p, o)).cuda(), triples).all() for o in bisimulation_[j])
                for s in tuple(bisimulation_[i])
            )
        ]
        quotient_triples.extend([(i, p, j) for p in filtered_predicates])

    quotient_triples = torch.tensor(quotient_triples, dtype=torch.long)

    return quotient_triples, bisimulation
