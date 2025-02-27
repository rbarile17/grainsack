import torch

import networkx as nx

from bispy import compute_maximum_bisimulation


def simulation(kg, entity, triples):
    """Compute the simulation of a knowledge graph."""
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

def preprocess(kg, multigraph):
    """Preprocess the multigraph to a directed graph."""
    digraph = nx.DiGraph()

    for s, o, data in multigraph.edges(data=True):
        p = data["label"]

        if not digraph.has_node(s):
            digraph.add_node(s, **{"label": kg.id_to_entity[s]})
        if not digraph.has_node(o):
            digraph.add_node(o, **{"label": kg.id_to_entity[o]})
        po_node = (p, o)
        if not digraph.has_node(po_node):
            po_label = f"{p}_{kg.id_to_entity[o]}"
            digraph.add_node(po_node, **{"label": po_label})

        digraph.add_edge(s, po_node)

    return digraph

def bisimulation(kg, triples):
    """Compute the bisimulation of a knowledge graph."""
    edges = [(s.item(), o.item(), {"label": p.item()}) for s, p, o in triples]
    graph = nx.MultiDiGraph(edges)
    
    digraph = preprocess(kg, graph)
    partition = kg.get_partition(triples)
    partition = [frozenset(cl.tolist()) for cl in partition]

    for node in digraph.nodes():
        if isinstance(node, tuple):
            partition.append(frozenset({node}))

    bisimulation = compute_maximum_bisimulation(digraph, partition)
    bisimulation = [frozenset(cl) for cl in bisimulation]

    contain_tuples = lambda cl: any(isinstance(node, tuple) for node in cl)
    bisimulation = [cl for cl in bisimulation if not contain_tuples(cl)]
    bisimulation_ = [list(cl) for cl in bisimulation]
    bisimulation = [torch.tensor(cl).cuda() for cl in bisimulation_]

    n_classes = len(bisimulation)

    subjects = triples[:, 0]
    objects = triples[:, 2]

    subject_masks = [torch.isin(subjects, cl) for cl in bisimulation]
    object_masks = [torch.isin(objects, cl) for cl in bisimulation]
    pairs = [(i, j) for i in range(n_classes) for j in range(n_classes)]

    quotient_triples = []
    for i, j in pairs:
        predicates = torch.unique(triples[subject_masks[i] & object_masks[j]][:, 1])

        qs, qo = tuple(bisimulation_[i]), tuple(bisimulation_[j])
        filtered_predicates = []
        for p in predicates:
            if all(any(torch.isin(torch.tensor((s, p, o)).cuda(), triples).all() for o in qo) for s in qs):
                filtered_predicates.append(p)
        quotient_triples.extend([(i, p, j) for p in filtered_predicates])

    quotient_triples = torch.tensor(quotient_triples, dtype=torch.long)

    return quotient_triples, bisimulation