"""Compute the simulation of a knowledge graph."""

import torch


def simulation_summary(kg, triples):
    """Compute the simulation of a (sub-)graph"""
    partition = kg.get_partition(triples)

    subjects = triples[:, 0]
    predicates = triples[:, 1]
    objects = triples[:, 2]

    max_ent = torch.max(torch.stack([subjects.max(), objects.max()])).item()
    entity2class = torch.full((max_ent + 1,), -1, dtype=torch.int32).cuda()

    for cid, ents in enumerate(partition):
        entity2class[ents] = cid

    cs = entity2class[subjects].to(torch.int64)
    co = entity2class[objects].to(torch.int64)

    valid = (cs >= 0) & (co >= 0)

    q = torch.stack([cs[valid], predicates[valid], co[valid]], dim=1)
    q = torch.unique(q, dim=0)

    return q, partition
