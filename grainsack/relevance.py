"""Functions for computing the relevance of statements with respect to predictions."""

import torch
from torch.linalg import inv


from grainsack import NECESSARY, SUFFICIENT
from grainsack.kge_lp import MODEL_REGISTRY

from grainsack.kge_lp import rank, train_kge_model


def replicate_prediction(replication_entities, prediction):
    """Preprocess the prediction for the relevance computation."""
    n_replications = replication_entities.size(0)

    replicated_prediction = prediction.clone().unsqueeze(0).repeat(n_replications, 1)
    replicated_prediction[:, 0] = replication_entities

    return replicated_prediction


def estimate_rank_variation(
    kg, kge_model, kge_config, fuse, replication_entities, prediction, statements, original_statements, partition, mode
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
    n_replications = replication_entities.size(0)
    n_statements = statements.size(0)

    kelpie_entities = torch.arange(n_replications * n_statements).cuda().view(n_replications, n_statements) + kg.num_entities

    triples = kg.training_triples.clone()
    triples = triples.unsqueeze(0).unsqueeze(0).repeat(n_replications, n_statements, 1, 1)

    mask = triples[..., [0, 2]] == replication_entities.view(n_replications, 1, 1, 1)
    triples[..., [0, 2]] = torch.where(mask, kelpie_entities.unsqueeze(-1).unsqueeze(-1), triples[..., [0, 2]])

    triples = triples[mask.any(dim=-1)]

    replicated_prediction = prediction.clone().unsqueeze(0).repeat(n_replications, 1)
    replicated_prediction[:, 0] = replication_entities
    kelpie_prediction = replicated_prediction.unsqueeze(1).repeat(1, n_statements, 1)
    kelpie_prediction[:, :, 0] = kelpie_entities

    kelpie_model_class = MODEL_REGISTRY[kge_model._get_name()]["kelpie_class"]
    base_model = kelpie_model_class(kg.training, kge_model, kge_config["model_kwargs"], n_replications, n_statements).cuda()
    train_kge_model(base_model, triples, **kge_config)
    base_results = rank(base_model, kelpie_prediction.reshape(-1, 3), filtr=[triples])
    base_results = base_results.reshape(n_replications, n_statements)

    pt_model = kelpie_model_class(kg.training, kge_model, kge_config["model_kwargs"], n_replications, n_statements).cuda()

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

    train_kge_model(pt_model, triples, **kge_config)
    pt_results = rank(pt_model, kelpie_prediction.reshape(-1, 3), filtr=[triples])
    pt_results = pt_results.reshape(n_replications, n_statements)

    if mode == NECESSARY:
        relevance = base_results - pt_results
    elif mode == SUFFICIENT:
        relevance = pt_results - base_results
    else:
        raise ValueError(f"Unknown mode: {mode}")

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


def dp_relevance(model, lr, aggregate, replication_entities, prediction, statements, lambd=1):
    """Measure for each statement the relevance wrt the prediction based on embedding perturbation

    Let <s, p, o> be the prediction, compute the perturbed entity embedding s_
    by shifting s based on the gradient and measure the relevance of a statement (s, q, e)
    as the difference between the score of (s, q, e) and the score of (s_, q, e).

    """

    replicated_prediction = replicate_prediction(replication_entities, prediction)

    n_replications = replicated_prediction.size(0)
    n_statements = statements.size(0)

    model.cuda()
    model.eval()

    gradients = []
    for i in range(n_replications):
        lhs = model.entity_representations[0](replicated_prediction[i, 0].reshape(-1)).detach()
        rel = model.relation_representations[0](replicated_prediction[i, 1].reshape(-1)).detach()
        rhs = model.entity_representations[0](replicated_prediction[i, 2].reshape(-1)).detach()

        lhs.requires_grad = True

        score = model.interaction(lhs, rel, rhs)
        score = score.sum()
        score.backward()
        gradient = lhs.grad
        lhs.grad = None

        gradients.append(gradient.cuda().detach())

    perturbed_entity_embeddings = [
        model.entity_representations[0](replicated_prediction[i, 0]).detach() for i in range(n_replications)
    ]
    perturbed_entity_embeddings = torch.vstack(
        [perturbed_entity_embeddings[i] - lr * gradients[i] for i in range(n_replications)]
    )

    mask = statements[:, :, 0] == replicated_prediction[:, 0].unsqueeze(-1)
    mask = mask.reshape(-1)

    statements = statements.reshape(-1, 3)
    lhs = model.entity_representations[0](statements[:, 0])
    rhs = model.entity_representations[0](statements[:, 2])
    rel = model.relation_representations[0](statements[:, 1])
    embedding_dim = lhs.size(1)

    original_scores = model.interaction(lhs, rel, rhs)

    lhs[mask] = perturbed_entity_embeddings.unsqueeze(1).repeat(1, n_statements, 1).reshape(-1, embedding_dim)[mask]

    perturbed_scores = model.interaction(lhs, rel, rhs)

    relevance = aggregate(original_scores, lambd * perturbed_scores)
    relevance = relevance.reshape(n_replications, n_statements).mean(dim=0)

    return relevance


def criage_relevance(kg, model, replication_entities, prediction, statements, mode):
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

    n_replications = replication_entities.size(0)
    n_statements = statements.size(0)

    statements = statements.squeeze(1).unsqueeze(0).repeat(n_replications, 1, 1)
    subject_mask = statements[:, :, 2] == prediction[0]
    object_mask = statements[:, :, 2] == prediction[2]

    statements[:, :, 2] = replication_entities.view(n_replications, 1)

    criage_prediction = prediction.unsqueeze(0).repeat(n_replications, 1)
    criage_prediction = criage_prediction.unsqueeze(1).repeat(1, n_statements, 1)

    criage_prediction[:, :, 0] = torch.where(
        subject_mask, replication_entities.view(n_replications, 1), criage_prediction[:, :, 0]
    )
    criage_prediction[:, :, 2] = torch.where(
        object_mask, replication_entities.view(n_replications, 1), criage_prediction[:, :, 2]
    )

    subjects = criage_prediction[..., 0].clone()
    objects = criage_prediction[..., 2].clone()
    criage_prediction[..., 0] = torch.where(object_mask, objects, subjects)
    criage_prediction[..., 2] = torch.where(object_mask, subjects, objects)

    z_pred = model.criage_first_step(criage_prediction.reshape(-1, 3)).detach().reshape(n_replications, n_statements, -1)
    z_candidates = model.criage_first_step(statements.reshape(-1, 3)).detach()

    hessian_matrix = [hessian(model, e, kg.training_triples[kg.training_triples[:, 2] == e]) for e in replication_entities]
    hessian_matrix = torch.stack(hessian_matrix)

    z_candidates = z_candidates.reshape(n_replications, n_statements, -1)

    entity_embedding = model.entity_representations[0](replication_entities).detach()

    x_2 = torch.bmm(entity_embedding.unsqueeze(1), z_candidates.permute(0, 2, 1)).squeeze(1)
    sig_tri = torch.sigmoid(x_2)
    sig_tri = sig_tri * (1 - sig_tri)
    sig_tri = sig_tri.unsqueeze(-1).unsqueeze(-1)

    dot = torch.bmm(z_candidates.permute(0, 2, 1), z_candidates).unsqueeze(1)

    m = hessian_matrix.unsqueeze(1) + sig_tri * dot
    m = inv(m)
    dot = (1 - sig_tri).squeeze(-1) * torch.matmul(z_candidates.unsqueeze(2), m).squeeze(2)

    relevance = torch.matmul(z_pred.unsqueeze(2), dot.unsqueeze(-1)).squeeze(-1)

    relevance = relevance.reshape(n_replications, n_statements).mean(dim=0)

    if relevance.dtype == torch.complex64 or relevance.dtype == torch.complex128:
        relevance = torch.abs(relevance)

    if mode == NECESSARY:
        return relevance
    elif mode == SUFFICIENT:
        return -relevance
    else:
        raise ValueError(f"Unknown mode: {mode}")
