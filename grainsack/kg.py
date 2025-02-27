from ast import literal_eval
from typing import Dict, List

import networkx as nx
import pandas as pd
import torch
from pykeen.datasets import get_dataset
from pykeen.triples.triples_factory import TriplesFactory

from . import KGS_PATH


class KG:
    def __init__(self, kg: str) -> None:
        self.name = kg

        # rdflib_graph = Graph()
        # rdflib_graph.parse(DATA_PATH / self.name / "train_rdf.ttl", format="nt")

        training = KGS_PATH / self.name / "train.txt"
        testing = KGS_PATH / self.name / "test.txt"
        validation = KGS_PATH / self.name / "valid.txt"

        self._kg = get_dataset(training=training, testing=testing, validation=validation)

        self.id_to_entity: Dict = {v: k for k, v in self._kg.entity_to_id.items()}
        self.id_to_relation: Dict = {v: k for k, v in self._kg.relation_to_id.items()}

        # entity_types = pd.read_csv(KGS_PATH / self.name / "reasoned" / "entities.csv", converters={"classes": literal_eval})
        # entity_types["entity"] = entity_types["entity"].map(self.entity_to_id.get)
        # entity_types["classes_str"] = entity_types["classes"].map(", ".join)

        # self.entity_types = entity_types

        self.nx_graph = nx.MultiGraph()
        self.nx_graph.add_nodes_from(list(self.id_to_entity.keys()))
        self.nx_graph.add_edges_from([(h.item(), t.item()) for h, _, t in self.training_triples])

    @property
    def training(self) -> TriplesFactory:
        return self._kg.training

    @property
    def training_triples(self) -> torch.Tensor:
        return self._kg.training.mapped_triples.cuda()

    @property
    def validation(self) -> TriplesFactory:
        return self._kg.validation

    @property
    def validation_triples(self) -> torch.Tensor:
        return self._kg.validation.mapped_triples.cuda()

    @property
    def testing(self) -> TriplesFactory:
        return self._kg.testing

    @property
    def testing_triples(self) -> torch.Tensor:
        return self._kg.testing.mapped_triples

    @property
    def num_entities(self) -> int:
        return self._kg.num_entities

    @property
    def num_relations(self) -> int:
        return self._kg.num_relations

    @property
    def entity_to_id(self):
        return self._kg.entity_to_id

    @property
    def relation_to_id(self):
        return self._kg.relation_to_id

    def label_triple(self, triple):
        s, p, o = triple
        return (self.id_to_entity[s], self.id_to_relation[p], self.id_to_entity[o])

    def label_triples(self, triples):
        return [self.label_triple(ids_triple) for ids_triple in triples]

    def id_triple(self, labeled_triple):
        s, p, o = labeled_triple
        return (self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o])

    def id_triples(self, labeled_triples):
        return [self.id_triple(t) for t in labeled_triples]

    def get_partition(self, triples):
        nodes = set([s.item() for s, _, _ in triples] + [o.item() for _, _, o in triples])
        entity_types = self.entity_types[self.entity_types.entity.isin(nodes)]
        partition = entity_types.groupby("classes_str")["entity"].apply(list)
        partition = partition.to_dict()
        partition = [torch.tensor(p).cuda() for p in partition.values()]

        return partition
