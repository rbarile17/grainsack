"""Kelpie model for PyKEEN."""

import numpy as np
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import BasicNegativeSampler
from pykeen.models import Model, TransE, ComplEx, ConvE
from pykeen.nn import Embedding
from torch import FloatTensor, LongTensor
from torch.nn.init import xavier_uniform_, normal_
from torch.nn import ModuleList

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pykeen.training import SLCWATrainingLoop
from pykeen.triples.triples_factory import CoreTriplesFactory


class NoResetEmbedding(torch.nn.Embedding):
    """A embedding layer that does not reset its parameters."""

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
    """Embedding layer for the Kelpie model."""

    def __init__(self, weight, max_id: int, shape: int, n_statements: int, dtype=torch.float):
        """Initialize the Kelpie embedding.
        Args:
            weight: The weight of the embedding.
            max_id: The maximum id of the embedding.
            shape: The shape of the embedding.
            n_statements: The number of statements.
        """
        super().__init__(max_id=max_id, shape=shape, dtype=dtype)

        if self.is_complex:
            self.embedding_dim = shape[0] * 2
        elif shape == ():
            self.embedding_dim = 1
        else:
            self.embedding_dim = shape[0]

        self._embeddings = NoResetEmbedding(weight, num_embeddings=max_id, embedding_dim=self.embedding_dim)
        if self.embedding_dim == 1:
            self._embeddings.weight.data = self._embeddings.weight.data.unsqueeze(-1)
        self.kelpie_embeddings = ModuleList(
            [NoResetEmbedding(num_embeddings=1, embedding_dim=self.embedding_dim) for _ in range(n_statements)]
        )
        self._embeddings.weight.requires_grad = False
        for kelpie_embedding in self.kelpie_embeddings:
            if self.is_complex:
                kelpie_embedding.weight = normal_(kelpie_embedding.weight)
            else:
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

            x = torch.zeros((indices.size(0), self.embedding_dim)).to(self.device)

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


class KelpieRelationEmbeddings(Embedding):
    def __init__(self, weight, max_id: int, shape: int):
        super().__init__(max_id=max_id, shape=shape)

        self._embeddings = NoResetEmbedding(weight, num_embeddings=max_id, embedding_dim=shape[0])
        self._embeddings.weight.requires_grad = False

    def reset_parameters(self):
        return


class KelpieTransE(TransE):
    """Kelpie TransE model."""

    def __init__(self, triples_factory, model: Model, config, n_statements):
        """Initialize the Kelpie TransE model.
        Args:
            triples_factory: The triples factory.
            model: The base model.
            config: The configuration.
            n_conversions: The number of conversions.
            n_candidates: The number of candidates.
        """
        super().__init__(triples_factory=triples_factory, **config, random_seed=42)

        weight = model.entity_representations[0]()
        max_id = self.entity_representations[0].max_id
        shape = self.entity_representations[0].shape
        self.entity_representations[0] = KelpieEmbedding(weight, max_id, shape, n_statements)

        weight = model.relation_representations[0]()
        max_id = self.relation_representations[0].max_id
        shape = self.relation_representations[0].shape
        self.relation_representations[0] = KelpieRelationEmbeddings(weight, max_id, shape)


class CustomComplEx(ComplEx):
    """Custom ComplEx model."""

    def criage_first_step(self, triples):
        """Get the first step of the query."""
        lhs = self.entity_representations[0](triples[:, 0])
        rel = self.relation_representations[0](triples[:, 1])

        return lhs * rel

    def _get_name(self):
        return "ComplEx"


class CustomConvE(ConvE):
    """Custom ConvE model."""

    def criage_first_step(self, triples):
        """Get the first step of the query."""

        s, p, _ = self._get_representations(h=triples[..., 0], r=triples[..., 1], t=None, mode=None)

        x = torch.cat(
            torch.broadcast_tensors(
                s.view(
                    *s.shape[:-1],
                    self.interaction.shape_info.input_channels,
                    self.interaction.shape_info.image_height,
                    self.interaction.shape_info.image_width,
                ),
                p.view(
                    *p.shape[:-1],
                    self.interaction.shape_info.input_channels,
                    self.interaction.shape_info.image_height,
                    self.interaction.shape_info.image_width,
                ),
            ),
            dim=-2,
        )
        prefix_shape = x.shape[:-3]
        x = x.view(
            -1,
            self.interaction.shape_info.input_channels,
            2 * self.interaction.shape_info.image_height,
            self.interaction.shape_info.image_width,
        )

        x = self.interaction.hr2d(x)

        x = x.view(-1, self.interaction.shape_info.num_in_features)
        x = self.interaction.hr1d(x)

        x = x.view(*prefix_shape, s.shape[-1])

        return x

    def _get_name(self):
        return "ConvE"


class KelpieComplEx(CustomComplEx):
    """Kelpie ComplEx model."""

    def __init__(self, triples_factory, model, config, n_statements):
        """Initialize the Kelpie ComplEx model."""
        super().__init__(triples_factory=triples_factory, **config, random_seed=42)

        weight = model.entity_representations[0]()
        weight = torch.view_as_real(weight).reshape(weight.shape[0], -1)
        max_id = self.entity_representations[0].max_id
        shape = self.entity_representations[0].shape
        self.entity_representations[0] = KelpieEmbedding(weight, max_id, shape, n_statements, dtype=torch.cfloat)

        weight = model.relation_representations[0]()
        max_id = self.relation_representations[0].max_id
        shape = self.relation_representations[0].shape
        self.relation_representations[0] = KelpieRelationEmbeddings(weight, max_id, shape)


class KelpieConvE(CustomConvE):
    """Kelpie ConvE model."""

    def __init__(self, triples_factory, model, config, n_statements):
        """Initialize the Kelpie ConvE model."""
        super().__init__(triples_factory=triples_factory, **config, random_seed=42, apply_batch_normalization=False)

        weight = model.entity_representations[0]()
        max_id = self.entity_representations[0].max_id
        shape = self.entity_representations[0].shape
        self.entity_representations[0] = KelpieEmbedding(weight, max_id, shape, n_statements)

        weight = model.entity_representations[1]()
        max_id = self.entity_representations[1].max_id
        shape = self.entity_representations[1].shape
        self.entity_representations[1] = KelpieEmbedding(weight, max_id, shape, n_statements)

        weight = model.relation_representations[0]()
        max_id = self.relation_representations[0].max_id
        shape = self.relation_representations[0].shape
        self.relation_representations[0] = KelpieRelationEmbeddings(weight, max_id, shape)

        self.interaction.requires_grad_(False)


MODEL_REGISTRY = {
    "TransE": {"class": TransE, "epochs": 1, "batch_size": 32, "kelpie_class": KelpieTransE},
    "ComplEx": {"class": CustomComplEx, "epochs": 1, "batch_size": 32, "kelpie_class": KelpieComplEx},
    "ConvE": {"class": CustomConvE, "epochs": 1000, "batch_size": 8192, "kelpie_class": KelpieConvE},
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


def train_kge_model(kge_model, triples, **kge_config):
    optimizer = Adam(params=kge_model.get_grad_params(), **kge_config["optimizer_kwargs"])
    lr_scheduler = ExponentialLR(optimizer=optimizer, **kge_config["lr_scheduler_kwargs"])
    training_triples = CoreTriplesFactory.create(triples.cpu())
    negative_sampler = BasicNegativeSampler(
        **kge_config["negative_sampler_kwargs"], mapped_triples=training_triples.mapped_triples
    )
    trainer = SLCWATrainingLoop(
        model=kge_model,
        triples_factory=training_triples,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        negative_sampler=negative_sampler,
    )
    trainer.train(triples_factory=training_triples, **kge_config["training_kwargs"], use_tqdm=False)
