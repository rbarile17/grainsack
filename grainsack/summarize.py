import torch

import networkx as nx

from bispy import compute_maximum_bisimulation


def simulation(kg, entity, triples):
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

def preprocess(self, multigraph):
    digraph = nx.DiGraph()

    for s, o, data in multigraph.edges(data=True):
        p = data["label"]

        if not digraph.has_node(s):
            digraph.add_node(s, **{"label": self.dataset.id_to_entity[s]})
        if not digraph.has_node(o):
            digraph.add_node(o, **{"label": self.dataset.id_to_entity[o]})
        po_node = (p, o)
        if not digraph.has_node(po_node):
            po_label = f"{p}_{self.dataset.id_to_entity[o]}"
            digraph.add_node(po_node, **{"label": po_label})

        digraph.add_edge(s, po_node)

    return digraph

def bisimulation(self, entity, triples):
    subgraph = self.dataset.get_subgraph(entity, triples=triples)
    digraph = self.preprocess(subgraph)
    partition = self.dataset.get_partition_sub(subgraph)

    n_classes = len(partition)

    for node in digraph.nodes():
        if is_tuple(node):
            partition.append(frozenset({node}))

    bisimulation = compute_maximum_bisimulation(digraph, partition)
    bisimulation = [frozenset(cl) for cl in bisimulation]
    bisimulation = [cl for cl in bisimulation if not contain_tuples(cl)]

    subjects = triples[:, 0]
    objects = triples[:, 2]

    partition_ = [list(cl) for cl in partition]

    subject_masks = [torch.isin(subjects, cl) for cl in partition]
    object_masks = [torch.isin(objects, cl) for cl in partition]
    pairs = [(i, j) for i in range(n_classes) for j in range(n_classes)]

    q_triples = []
    for pair in pairs:
        i, j = pair

        predicates = torch.unique(triples[subject_masks[i] & object_masks[j]][:, 1])

        qs, qo = tuple(part_map[i]), tuple(part_map[j])
        filtered_predicates = []
        for p in predicates:
            if all(any((s, p, o) in triples for o in qo) for s in qs):
                filtered_predicates.append(p)
        q_triples.extend([(qs, p.item(), qo) for p in filtered_predicates])

    return q_triples