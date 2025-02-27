"""Functions for computing the relevance of statements with respect to a prediction."""

import torch
from pykeen.models.base import Model
from pykeen.training import SLCWATrainingLoop
from pykeen.triples.triples_factory import CoreTriplesFactory
from torch.linalg import inv
from torch.optim import Adam

from grainsack.kge_lp import KelpieTransE

from .kge_lp import rank



def replicate_prediction(conversion_entities, prediction):
    """Preprocess the prediction for the relevance computation."""
    n_conversions = conversion_entities.size(0)

    replicated_prediction = prediction.clone().unsqueeze(0).repeat(n_conversions, 1)
    replicated_prediction[:, 0] = conversion_entities

    return replicated_prediction

def estimate_rank_variation(
    kg, model, model_kwargs, fuse, replication_entities, prediction, statements, original_statements, partition
) -> float:
    """Estimate the rank variation of a prediction due to each statement.

    Estimate the rank variation of a prediction due to a statement by
    averaging the rank variations of the replications of the prediction.
    The rank variation of each replication due to a statement is the aggregation (e.g., sum) of its ranks before
    and after modifying the KG according to the statement and post-training the model.

    Args:
        kg: the KG.
        model: the model for computing the rank of the prediction.
        model_kwargs: the model hyper-parameters.
        modify: the function for modifying the KG according to the statement set.
        replication_entities: set of entities over which the prediction is repeated.
        prediction: the prediction to be explained.
        statements: the statements for which the rank variation is computed.

    Returns:
        The rank variations.
    """
    replicated_prediction = replicate_prediction(replication_entities, prediction)
    n_replications = replication_entities.size(0)
    n_statements = statements.size(0)

    kelpie_entities = torch.arange(0, n_replications * n_statements).cuda()
    kelpie_entities = kelpie_entities.reshape(n_replications, n_statements)
    kelpie_entities = kelpie_entities + kg.num_entities

    triples = kg.training_triples.clone()
    subject_mask = triples[:, 0].unsqueeze(-1) == replication_entities.unsqueeze(0)
    object_mask = triples[:, 2].unsqueeze(-1) == replication_entities.unsqueeze(0)
    mask = subject_mask | object_mask
    triples = triples.unsqueeze(0) * mask.T.unsqueeze(-1)

    triples = triples.unsqueeze(1).repeat(1, n_statements, 1, 1)
    mask = triples[:, :, :, 0] == replication_entities.unsqueeze(-1).unsqueeze(-1)
    triples[:, :, :, 0] = torch.where(mask, kelpie_entities.unsqueeze(-1), triples[:, :, :, 0])
    mask = triples[:, :, :, 2] == replication_entities.unsqueeze(-1).unsqueeze(-1)
    triples[:, :, :, 2] = torch.where(mask, kelpie_entities.unsqueeze(-1), triples[:, :, :, 2])
    triples = triples.reshape(n_statements, -1, 3)
    mask = (triples == torch.zeros(3).cuda()).all(dim=-1)
    triples = triples[~mask]

    kelpie_prediction = replicated_prediction.unsqueeze(1).repeat(1, n_statements, 1)
    kelpie_prediction[:, :, 0] = kelpie_entities

    base_model = KelpieTransE(kg.training, model, model_kwargs, n_replications, n_statements)
    base_model.cuda()
    base_results = rank(base_model, kelpie_prediction.reshape(-1, 3), filtr=[triples])
    base_results = base_results.reshape(n_replications, n_statements)

    pt_model = KelpieTransE(kg.training, model, model_kwargs, n_replications, n_statements)
    pt_model = post_train(
        pt_model, fuse, prediction, triples, replication_entities, kelpie_entities, statements, original_statements, partition
    )
    pt_results = rank(pt_model, kelpie_prediction.reshape(-1, 3), filtr=[triples])
    pt_results = pt_results.reshape(n_replications, n_statements)

    relevance = pt_results - base_results
    relevance = relevance.mean(dim=0)

    return relevance


def add_statements(triples, statements):
    """Add the statements to the triples."""
    statements = statements[~((statements == torch.zeros(3).cuda()).all(dim=-1))]
    triples = torch.cat([triples, statements], dim=0)


def remove_statements(triples, statements):
    """Remove the statements from the triples."""
    statements = statements[~((statements == torch.zeros(3).cuda()).all(dim=-1))]
    mask = (triples.unsqueeze(1) == statements.unsqueeze(0)).all(dim=-1)
    mask = mask.any(dim=-1)
    triples = triples[~mask]


def post_train(
    model, fuse, prediction, triples, replication_entities, kelpie_entities, statements, original_statements, partition
) -> Model:
    """Post-train the model on the modified KG.

    Modify the KG according to the statement set and post-train the model on the modified KG.
    Post-training is done by creating a new entity embedding for each statement set,
    freezing the existing entity and relation embeddings of the model,
    and training the model on the modified KG.

    Args:
        model: the model to be post-trained.
        modify: the function fusing the KG and the statement sets.
        kelpie_kg: the modified KG.
        statement_sets: the statement sets.
    """

    model.cuda()

    n_statements = statements.size(0)

    mapped_statements = []
    for statement in statements:
        statement = statement.reshape(-1, 3)
        mapped_statement = []
        for i, p, j in statement:
            if partition == []:
                mapped_statement.append((i, p, j))
            else:
                mapped_statement.extend([(s, p, o) for s in partition[i] for o in partition[j]])
        mapped_statements.append(torch.tensor(mapped_statement).cuda())

    masks = [(s.unsqueeze(1) == original_statements).all(dim=-1).any(dim=1) for s in mapped_statements]
    mapped_statements = [mapped_statements[i][masks[i]] for i in range(n_statements)]

    n_replications = replication_entities.size(0)

    mapped_statements_replications = []
    for m in mapped_statements:
        m = m.unsqueeze(0).repeat(n_replications, 1, 1)
        mask = m[:, :, 0] == prediction[0]
        m[:, :, 0] = torch.where(mask, replication_entities.view(n_replications, 1), m[:, :, 0])
        mask = m[:, :, 2] == prediction[0]
        m[:, :, 2] = torch.where(mask, replication_entities.view(n_replications, 1), m[:, :, 2])
        mapped_statements_replications.append(m)

    original_entities = replication_entities.unsqueeze(-1)
    kelpie_statements = [m.clone() for m in mapped_statements_replications]

    for i in range(n_statements):
        mask = kelpie_statements[i][:, :, 0] == original_entities
        kelpie_statements[i][:, :, 0] = torch.where(mask, kelpie_entities[:, i].unsqueeze(-1), kelpie_statements[i][:, :, 0])
        mask = kelpie_statements[i][:, :, 2] == original_entities
        kelpie_statements[i][:, :, 2] = torch.where(mask, kelpie_entities[:, i].unsqueeze(-1), kelpie_statements[i][:, :, 2])

    kelpie_statements = torch.cat(kelpie_statements, dim=1).reshape(-1, 3)

    fuse(triples, kelpie_statements)

    optimizer = Adam(params=model.get_grad_params())

    training_triples = triples.cpu()

    post_triples = CoreTriplesFactory.create(training_triples)
    trainer = SLCWATrainingLoop(model=model, triples_factory=post_triples, optimizer=optimizer)

    trainer.train(triples_factory=post_triples, use_tqdm=False)

    return model

def get_gradients(model, prediction):
    """Get the gradients of the model with respect to the prediction."""
    n_conversions = prediction.size(0)
    gradients = []
    for i in range(n_conversions):
        lhs = model.entity_representations[0](prediction[i, 0].reshape(-1)).detach()
        rel = model.relation_representations[0](prediction[i, 1].reshape(-1)).detach()
        rhs = model.entity_representations[0](prediction[i, 2].reshape(-1)).detach()

        lhs.requires_grad = True

        score = model.interaction(lhs, rel, rhs)
        score = score.sum()
        score.backward()
        gradient = lhs.grad
        lhs.grad = None

        gradients.append(gradient.cuda().detach())

    return gradients


def dp_relevance(model, lr, aggregate, conversion_entities, prediction, statements, lambd=1):
    """Measure for each statement the relevance wrt the prediction based on embedding perturbation

    Let <s, p, o> be the prediction, compute the perturbed entity embedding s_ 
    by shifting s based on the gradient and measure the relevance of a statement (s, q, e)
    as the difference between the score of (s, q, e) and the score of (s_, q, e).

    """
    prediction, statements = replicate_prediction(conversion_entities, prediction)

    n_conversions = prediction.size(0)
    n_candidates = statements.size(1)

    model.cuda()
    model.eval()

    statements = statements.squeeze(2)

    gradients = get_gradients(model, prediction)

    perturbed_entity_embeddings = [model.entity_representations[0](prediction[i, 0]).detach() for i in range(n_conversions)]
    perturbed_entity_embeddings = torch.vstack([perturbed_entity_embeddings[i] - lr * gradients[i] for i in range(n_conversions)])

    mask = statements[:, :, 0] == prediction[:, 0].unsqueeze(-1)

    statements = statements.reshape(-1, 3)
    lhs = model.entity_representations[0](statements[:, 0])
    rhs = model.entity_representations[0](statements[:, 2])
    rel = model.relation_representations[0](statements[:, 1])

    mask = mask.reshape(-1)

    original_scores = model.interaction(lhs, rel, rhs)

    embedding_dim = lhs.size(1)

    lhs[mask] = perturbed_entity_embeddings.unsqueeze(1).repeat(1, n_candidates, 1).reshape(-1, embedding_dim)[mask]

    perturbed_scores = model.interaction(lhs, rel, rhs)

    relevance = aggregate(original_scores, lambd * perturbed_scores)
    relevance = relevance.reshape(n_conversions, n_candidates).mean(dim=0)

    return relevance


def criage_relevance(kg, model, conversion_entities, prediction, candidates):
    """Measure the relevance of each statement based on influence functions."""

    def hessian(model, entities, triples):
        lhs = model.entity_representations[0](triples[:, 0]).detach()
        rel = model.relation_representations[0](triples[:, 1]).detach()

        x = lhs * rel
        x_2 = torch.matmul(model.entity_representations[0](entities).detach(), x.T)

        sig = torch.sigmoid(x_2)
        sig = sig * (1 - sig)
        hessian_matrix = sig.sum() * torch.matmul(x.T, x)

        return hessian_matrix

    model.cuda()
    model.eval()

    n_conversions = conversion_entities.size(0)
    n_candidates = candidates.size(0)

    candidates = candidates.unsqueeze(0).repeat(n_conversions, 1, 1)
    subject_mask = candidates[:, :, 2] == prediction[0]
    object_mask = candidates[:, :, 2] == prediction[2]

    candidates[:, :, 2] = conversion_entities.view(n_conversions, 1)

    criage_prediction = prediction.unsqueeze(0).repeat(n_conversions, 1)
    criage_prediction = criage_prediction.unsqueeze(1).repeat(1, n_candidates, 1)

    criage_prediction[:, :, 0] = torch.where(subject_mask, conversion_entities.view(n_conversions, 1), criage_prediction[:, :, 0])
    criage_prediction[:, :, 2] = torch.where(object_mask, conversion_entities.view(n_conversions, 1), criage_prediction[:, :, 2])

    subjects = criage_prediction[..., 0].clone()
    objects = criage_prediction[..., 2].clone()
    criage_prediction[..., 0] = torch.where(object_mask, objects, subjects)
    criage_prediction[..., 2] = torch.where(object_mask, subjects, objects)

    z_pred = model.criage_first_step(criage_prediction.reshape(-1, 3)).detach().reshape(n_conversions, n_candidates, -1)
    z_candidates = model.criage_first_step(candidates.reshape(-1, 3)).detach()

    hessian_matrix = [hessian(model, e, kg.training_triples[kg.training_triples[:, 2] == e]) for e in conversion_entities]
    hessian_matrix = torch.stack(hessian_matrix)

    z_candidates = z_candidates.reshape(n_conversions, n_candidates, -1)

    entity_embedding = model.entity_representations[0](conversion_entities).detach()

    x_2 = torch.bmm(entity_embedding.unsqueeze(1), z_candidates.permute(0, 2, 1)).squeeze(1)
    sig_tri = torch.sigmoid(x_2)
    sig_tri = sig_tri * (1 - sig_tri)
    sig_tri = sig_tri.unsqueeze(-1).unsqueeze(-1)

    dot = torch.bmm(z_candidates.permute(0, 2, 1), z_candidates).unsqueeze(1)

    m = hessian_matrix.unsqueeze(1) + sig_tri * dot
    m = inv(m)
    dot = (1 - sig_tri).squeeze(-1) * torch.matmul(z_candidates.unsqueeze(2), m).squeeze(2)

    relevance = torch.matmul(z_pred.unsqueeze(2), dot.unsqueeze(-1)).squeeze(-1)

    # alla sufficient non sta il meno
    return -relevance
