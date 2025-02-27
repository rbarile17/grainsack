"""Kelpie model for PyKEEN."""

import numpy as np
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import Model, TransE, ComplEx
from pykeen.nn import Embedding
from torch import FloatTensor, LongTensor
from torch.nn.init import xavier_uniform_
from torch.nn import ModuleList


class NoResetEmbedding(torch.nn.Embedding):
    """No reset embedding class.
    This class is used to create an embedding layer that does not reset its parameters.
    It is a subclass of the PyTorch Embedding class and is used to create an embedding layer
    that can be used in the Kelpie model.
    """
    def __init__(self, weight=None, **kwargs):
        """Initialize the NoResetEmbedding.
        Args:
            weight: The weight of the embedding.
            kwargs: Additional arguments for the embedding layer.
        """
        super().__init__(**kwargs)

        if weight is not None:
            self.weight.data = weight.data

    def reset_parameters(self):
        """Override the reset_parameters method to do nothing."""
        return


class KelpieEmbedding(Embedding):
    """Kelpie embedding class.
    This class is used to create an embedding layer for the Kelpie model.
    It is a subclass of the PyTorch Embedding class and is used to create an embedding layer
    that can be used in the Kelpie model.
    """
    def __init__(self, weight, max_id: int, shape: int, n_replications: int, n_candidates: int):
        """Initialize the Kelpie embedding.
        Args:
            weight: The weight of the embedding.
            max_id: The maximum id of the embedding.
            shape: The shape of the embedding.
            n_conversions: The number of conversions.
            n_candidates: The number of candidates.
        """
        super().__init__(max_id=max_id, shape=shape)

        self._embeddings = NoResetEmbedding(weight, num_embeddings=max_id, embedding_dim=shape[0])
        n_kelpie = n_candidates * n_replications
        self.kelpie_embeddings = ModuleList([NoResetEmbedding(num_embeddings=1, embedding_dim=shape[0]) for _ in range(n_kelpie)])
        self._embeddings.weight.requires_grad = False
        for kelpie_embedding in self.kelpie_embeddings:
            kelpie_embedding.weight = xavier_uniform_(kelpie_embedding.weight)
            kelpie_embedding.weight.requires_grad = True

    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:
        """Get the plain forward pass of the embedding.
        Args:
            indices: The indices to get the embedding for.
        Returns:
            The embedding for the indices.
        """
        if indices is None:
            prefix_shape = (self.max_id,)
            x = self._embeddings.weight
        else:
            prefix_shape = indices.shape

            x = torch.zeros((indices.size(0), self.shape[0])).to(self.device)

            mask = (indices >= self.max_id).squeeze()
            kelpie_indices = (indices[mask] - self.max_id).to(self.device)
            plain_indices = indices[~mask].squeeze().to(self.device)

            if kelpie_indices.numel() > 0:
                zero_index = torch.zeros(1, dtype=torch.long).to(self.device)
                x[mask] = torch.vstack([self.kelpie_embeddings[i](zero_index) for i in kelpie_indices])
            if plain_indices.numel() > 0:
                x[~mask] = self._embeddings(plain_indices)

        x = x.view(*prefix_shape, *self._shape)

        if self.is_complex:
            x = torch.view_as_complex(x)

        return x

    def reset_parameters(self):
        return


class KelpieTransE(TransE):
    """Kelpie TransE model."""
    def __init__(self, triples_factory, model: Model, config, n_conversions, n_candidates):
        """Initialize the Kelpie TransE model.
        Args:
            triples_factory: The triples factory.
            model: The base model.
            config: The configuration.
            n_conversions: The number of conversions.
            n_candidates: The number of candidates.
        """
        super().__init__(triples_factory=triples_factory, **config, random_seed=42)

        max_id = self.entity_representations[0].max_id
        shape = self.entity_representations[0].shape

        weight = model.entity_representations[0]()
        self.entity_representations[0] = KelpieEmbedding(weight, max_id, shape, n_conversions, n_candidates)


class CustomComplEx(ComplEx):
    """Custom ComplEx model."""
    def __init__(self, triples_factory, **kwargs):
        """Initialize the Custom ComplEx model."""
        super().__init__(triples_factory=triples_factory, **kwargs)
        self.dimension = self.entity_representations[0]._embeddings.weight.data.shape[1]
        self.real_dimension = self.dimension // 2

    def _get_queries(self, triples):
        """Get the queries for the model."""
        lhs = self.entity_representations[0](triples[:, 0])
        rel = self.relation_representations[0](triples[:, 1])

        lhs = (lhs[:, : self.real_dimension], lhs[:, self.real_dimension :])
        rel = (rel[:, : self.real_dimension], rel[:, self.real_dimension :])

        real = lhs[0] * rel[0] - lhs[1] * rel[1]
        im = lhs[0] * rel[1] + lhs[1] * rel[0]

        return torch.cat([real, im], 1)

    def criage_first_step(self, triples):
        """Get the first step of the query."""
        return self._get_queries(triples)

    def criage_last_step(self, x: torch.Tensor, rhs: torch.Tensor):
        """Get the last step of the query."""
        return x @ rhs

    def get_kelpie_class(self):
        """Get the Kelpie class."""
        return KelpieComplEx


class KelpieComplEx(CustomComplEx):
    """Kelpie ComplEx model."""
    def __init__(self, triples_factory, model, config, n_conversions, n_candidates):
        """Initialize the Kelpie ComplEx model."""
        super().__init__(triples_factory=triples_factory, **config, random_seed=42)

        max_id = self.entity_representations[0].max_id
        shape = self.entity_representations[0].shape

        weight = model.entity_representations[0]()
        self.entity_representations[0] = KelpieEmbedding(weight, max_id, shape, n_conversions, n_candidates)


MODEL_REGISTRY = {
    "TransE": {"class": TransE, "epochs": 1, "batch_size": 64},
    "ComplEx": {"class": CustomComplEx, "epochs": 1, "batch_size": 64},
    # "ConvE": {"class": CustomConvE, "epochs": 5, "batch_size": 65536},
}


def rank(model, triples, filtr):
    """Evaluate the model on the given triples.
    Args:
        model: The model to evaluate.
        triples: The triples to evaluate on.
        filter: The filter to apply.
    Returns:
        The ranks of the model.
    """
    evaluator = RankBasedEvaluator(clear_on_finalize=False)

    evaluator.evaluate(model, triples, targets=("tail",), additional_filter_triples=filtr, use_tqdm=False, batch_size=64)

    ranks = evaluator.ranks[("tail", "optimistic")]

    return torch.Tensor(np.concatenate(ranks))
