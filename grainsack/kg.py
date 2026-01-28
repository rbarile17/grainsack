"""Implementation of the KG class."""

from collections import defaultdict
from time import strftime
from typing import Dict

import networkx as nx
import pandas as pd
import torch
from pykeen.datasets import get_dataset
from pykeen.triples.triples_factory import TriplesFactory
from rdflib import RDF, Graph

from . import KGS_PATH


class KG:
    """Knowledge graph representation with entity/relation mappings and utilities.
    
    Loads a knowledge graph from TSV split files, creates ID mappings, parses
    the RDF representation, extracts entity types, and builds a NetworkX graph
    for topology-based operations.
    
    Attributes:
        name (str): Name of the knowledge graph.
        id_to_entity (dict): Mapping from entity IDs to entity URIs.
        id_to_relation (dict): Mapping from relation IDs to relation URIs.
        rdf_kg (rdflib.Graph): RDF representation of the knowledge graph.
        entity_types (pd.DataFrame): Entity type information.
        nx_graph (networkx.MultiGraph): NetworkX graph for path computations.
    """

    def __init__(self, kg: str, create_inverse_triples=False) -> None:
        """Initialize and load a knowledge graph.
        
        Args:
            kg (str): Name of the knowledge graph (directory name in kgs directory).
            create_inverse_triples (bool, optional): Whether to create inverse
                triples for the KG (required for ConvE). Defaults to False.
        """

        self.name = kg

        train = KGS_PATH / self.name / "abox/splits/train.tsv"
        test = KGS_PATH / self.name / "abox/splits/test.tsv"
        valid = KGS_PATH / self.name / "abox/splits/valid.tsv"

        print(f"Loading tsv dataset", strftime("%H:%M:%S"))

        dataset_kwargs = {"create_inverse_triples": create_inverse_triples}
        self._kg = get_dataset(training=train, testing=test, validation=valid, dataset_kwargs=dataset_kwargs)

        print("Building ID mappings", strftime("%H:%M:%S"))

        self.id_to_entity: Dict = {v: k for k, v in self._kg.entity_to_id.items()}
        self.id_to_relation: Dict = {v: k for k, v in self._kg.relation_to_id.items()}

        print(f"Parsing RDF KG for {self.name}", strftime("%H:%M:%S"))
        self.rdf_kg = Graph()
        self.rdf_kg.parse(KGS_PATH / self.name / "kg.owl")

        print(f"Extracting entity types for {self.name}", strftime("%H:%M:%S"))

        entity_types = defaultdict(list)
        for s, _, o in self.rdf_kg.triples((None, RDF.type, None)):
            entity_types[str(s)].append(str(o))

        self.entity_types = pd.DataFrame(
            [(e, "; ".join(classes)) for e, classes in entity_types.items()], columns=["entity", "classes_str"]
        )
        self.entity_types["entity"] = self.entity_types["entity"].map(self.entity_to_id.get)

        self.nx_graph = nx.MultiGraph()
        self.nx_graph.add_nodes_from(list(self.id_to_entity.keys()))
        self.nx_graph.add_edges_from([(h.item(), t.item()) for h, _, t in self.training_triples])

    @property
    def training(self) -> TriplesFactory:
        """Get the training triples factory.
        
        Returns:
            TriplesFactory: PyKEEN triples factory for training data.
        """
        return self._kg.training

    @property
    def training_triples(self) -> torch.Tensor:
        """Get the training triples as a GPU tensor.
        
        Returns:
            torch.Tensor: Training triples with integer IDs, shape (N, 3), on CUDA.
        """
        return self._kg.training.mapped_triples.cuda()

    @property
    def validation(self) -> TriplesFactory:
        """Get the validation triples factory.
        
        Returns:
            TriplesFactory: PyKEEN triples factory for validation data.
        """
        return self._kg.validation

    @property
    def validation_triples(self) -> torch.Tensor:
        """Get the validation triples as a GPU tensor.
        
        Returns:
            torch.Tensor: Validation triples with integer IDs, shape (N, 3), on CUDA.
        """
        return self._kg.validation.mapped_triples.cuda()

    @property
    def testing(self) -> TriplesFactory:
        """Get the testing triples factory.
        
        Returns:
            TriplesFactory: PyKEEN triples factory for test data.
        """
        return self._kg.testing

    @property
    def testing_triples(self) -> torch.Tensor:
        """
        Get the testing triples.
        :return: The testing triples.
        """
        return self._kg.testing.mapped_triples.cuda()

    @property
    def num_entities(self) -> int:
        """
        Get the number of entities in the knowledge graph.
        :return: The number of entities.
        """
        return self._kg.num_entities

    @property
    def num_relations(self) -> int:
        """
        Get the number of relations in the knowledge graph.
        :return: The number of relations.
        """
        return self._kg.num_relations

    @property
    def entity_to_id(self):
        """
        Get the mapping from entity names to IDs.
        :return: The mapping from entity names to IDs.
        """
        return self._kg.entity_to_id

    @property
    def relation_to_id(self):
        """
        Get the mapping from relation names to IDs.
        :return: The mapping from relation names to IDs.
        """
        return self._kg.relation_to_id

    def label_triple(self, triple):
        """
        Convert a triple from IDs to labels.
        :param triple: The triple to convert.
        :return: The labeled triple.
        """

        s, p, o = triple
        return (self.id_to_entity[s.item()], self.id_to_relation[p.item()], self.id_to_entity[o.item()])

    def label_triples(self, triples):
        """
        Convert a list of triples from IDs to labels.
        :param triples: The list of triples to convert.
        :return: The list of labeled triples.
        """
        return [self.label_triple(ids_triple) for ids_triple in triples]

    def id_triple(self, labeled_triple):
        """
        Convert a labeled triple to IDs.
        :param labeled_triple: The labeled triple to convert.
        :return: The triple with IDs.
        """
        s, p, o = labeled_triple
        return (self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o])

    def id_triples(self, labeled_triples):
        """
        Convert a list of labeled triples to IDs.
        :param labeled_triples: The list of labeled triples to be converted.
        :return: The list of triples with IDs.
        :rtype: list[tuple]
        """
        return [self.id_triple(t) for t in labeled_triples]

    def get_partition(self, triples):
        """
        Get the partition of the knowledge graph.
        :param triples: The triples to partition.
        :return: The partition of the knowledge graph.
        """
        nodes = set([s.item() for s, _, _ in triples] + [o.item() for _, _, o in triples])
        entity_types = self.entity_types[self.entity_types.entity.isin(nodes)]
        partition = entity_types.groupby("classes_str")["entity"].apply(list)
        partition = partition.to_dict()
        partition = [torch.tensor(p, dtype=torch.long).cuda() for p in partition.values()]

        return partition
