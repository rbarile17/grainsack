"""Functions for computing the relevance of statements with respect to predictions."""

import torch

from grainsack import CONVE
from grainsack.kge_lp import MODEL_REGISTRY, rank, train_kge_model


def estimate_rank_variation(kg, kge_model, kge_config, prediction, statements, original_statements, partition, operation=None):
    """Estimate the rank variation of a prediction due to each statement.

    Estimate the rank variation of a prediction due to a statement by
    averaging the rank variations of the replications of the prediction.
    The rank variation of each replication due to a statement is the aggregation (e.g., sum) of its ranks before
    and after modifying the KG according to the statement and post-training the model.

    Args:
        kg: the KG.
        kge_model: the model for computing the rank of the prediction.
        kge_config: the model hyper-parameters.
        prediction: the prediction to be explained.
        statements: the statements for which the rank variation is computed.

    Returns:
        The rank variations.
    """

    n = statements.size(0)

    mimics = torch.arange(n).cuda() + kg.num_entities

    triples = kg.training_triples.clone()

    incident_triples = triples[(triples[:, 0] == prediction[0]) | (triples[:, 2] == prediction[0])]
    m = incident_triples.size(0)

    mimic_triples = incident_triples.unsqueeze(0).expand(n, m, 3).clone()

    mask = mimic_triples[..., [0, 2]] == prediction[0]
    mimic_triples[..., [0, 2]] = torch.where(mask, mimics[:, None, None], mimic_triples[..., [0, 2]])
    mimic_triples = mimic_triples.reshape(-1, 3)

    mimic_prediction = prediction.unsqueeze(0).expand(n, -1).clone()
    mimic_prediction[:, 0] = mimics

    mimic_model_class = MODEL_REGISTRY[kge_model._get_name()]["kelpie_class"]

    base_model = mimic_model_class(kg.training, kge_model, kge_config["model_kwargs"], n).cuda()
    train_kge_model(base_model, mimic_triples, **kge_config)
    base_ranks = rank(base_model, mimic_prediction, filtr=[mimic_triples])

    mapped_stmts = []
    for stmt in statements:
        mapped_stmt = []
        for i, p, j in stmt:
            if partition == []:
                mapped_stmt.append((i, p, j))
            else:
                mapped_stmt.extend([(s, p, o) for s in partition[i] for o in partition[j]])
        mapped_stmts.append(torch.tensor(mapped_stmt).cuda())

    masks = [(s.unsqueeze(1) == original_statements).all(dim=-1).any(dim=1) for s in mapped_stmts]
    mapped_stmts = [mapped_stmts[i][masks[i]] for i in range(n)]

    mimic_stmts = [m.clone() for m in mapped_stmts]

    for i in range(n):
        mask = mimic_stmts[i][:, [0, 2]] == prediction[0]
        mimic_stmts[i][:, [0, 2]] = torch.where(mask, mimics[i], mimic_stmts[i][:, [0, 2]])

    mimic_stmts = torch.cat(mimic_stmts, dim=0).reshape(-1, 3)

    if operation == "remove":
        mimic_triples = mimic_triples[~((mimic_triples.unsqueeze(1) == mimic_stmts.unsqueeze(0)).all(dim=-1).any(dim=-1))]
    elif operation == "add":
        mimic_triples = torch.cat([mimic_triples, mimic_stmts], dim=0)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    if mimic_triples.size(0) == 0:
        return torch.zeros(n).cuda()

    pt_model = mimic_model_class(kg.training, kge_model, kge_config["model_kwargs"], n).cuda()
    train_kge_model(pt_model, mimic_triples, **kge_config)
    pt_ranks = rank(pt_model, mimic_prediction, filtr=[mimic_triples])

    return pt_ranks - base_ranks


def dp_relevance(model, lr, prediction, statements, lambd=1):
    """Compute statement relevance using data poisoning via embedding perturbation.
    
    For prediction <s, p, o>, computes perturbed entity embedding s' by shifting s
    based on the gradient. Relevance of statement (s, q, e) is measured as the
    difference between scores of (s, q, e) and (s', q, e).
    
    Args:
        model: KGE model with entity and relation embeddings.
        lr (float): Learning rate for perturbation step size.
        prediction (torch.Tensor): Prediction triple (subject, predicate, object).
        statements (torch.Tensor): Candidate statement triples to score.
        lambd (float, optional): Weighting factor for perturbed scores. Defaults to 1.
        
    Returns:
        torch.Tensor: Relevance scores for each statement.
    """

    n_statements = statements.size(0)

    model.cuda()
    model.eval()

    lhs = model.entity_representations[0](prediction[0]).detach()
    rel = model.relation_representations[0](prediction[1]).detach()
    if model._get_name() == "ConvE":
        rhs = (
            model.entity_representations[0](prediction[2].reshape(-1)).detach(),
            model.entity_representations[1](prediction[2].reshape(-1)).detach(),
        )
    else:
        rhs = model.entity_representations[0](prediction[2].reshape(-1)).detach()
    lhs.requires_grad = True

    score = model.interaction(lhs, rel, rhs)
    score.backward()
    gradient = lhs.grad
    lhs.grad = None

    perturbed_entity_embedding = model.entity_representations[0](prediction[0]).detach() - lr * gradient

    statements = statements.squeeze(1)
    mask = statements[:, 0] == prediction[0]

    lhs = model.entity_representations[0](statements[:, 0])
    rel = model.relation_representations[0](statements[:, 1])
    if model._get_name() == CONVE:
        rhs = (model.entity_representations[0](statements[:, 2]), model.entity_representations[1](statements[:, 2]))
    else:
        rhs = model.entity_representations[0](statements[:, 2])

    original_scores = model.interaction(lhs, rel, rhs)

    lhs[mask] = perturbed_entity_embedding.unsqueeze(0).repeat(n_statements, 1)[mask]

    perturbed_scores = model.interaction(lhs, rel, rhs)

    return original_scores + lambd * perturbed_scores


def criage_relevance(kg, model, prediction, statements):
    """Compute statement relevance using influence functions (CRIAGE method).
    
    Estimates the influence of adding each statement on the prediction score
    using Hessian-based influence functions from the CRIAGE explanation method.
    
    Args:
        kg (KG): Knowledge graph.
        model: KGE model with entity and relation embeddings.
        prediction (torch.Tensor): Prediction triple to explain.
        statements (torch.Tensor): Candidate statement triples to score.
        
    Returns:
        torch.Tensor: Influence scores for each statement.
    """

    def get_hessian(model, entity, triples, lam=1e-4):
        lhs = model.entity_representations[0](triples[:, 0]).detach()
        rel = model.relation_representations[0](triples[:, 1]).detach()

        x = lhs * rel
        entity_embedding = model.entity_representations[0](entity).detach()
        x_2 = x @ entity_embedding.squeeze()

        sig = torch.sigmoid(x_2)
        sig = sig * (1 - sig)

        Xw = x * sig.unsqueeze(-1)
        H = Xw.T @ x

        d = H.size(0)
        H = H + lam * torch.eye(d, device=H.device, dtype=H.dtype)
        return H

    model.cuda()
    model.eval()

    n_statements = statements.size(0)

    statements = statements.squeeze(1)
    subject_mask = statements[:, 2] == prediction[0]
    object_mask = statements[:, 2] == prediction[2]

    criage_prediction = prediction.unsqueeze(0).repeat(n_statements, 1).clone()
    criage_prediction[subject_mask] = statements[subject_mask][:, [2, 1, 0]]

    hessian_head = get_hessian(model, prediction[0], kg.training_triples[kg.training_triples[:, 2] == prediction[0]])
    hessian_tail = get_hessian(model, prediction[0], kg.training_triples[kg.training_triples[:, 2] == prediction[0]])

    z_pred = model.criage_first_step(criage_prediction[subject_mask]).detach()
    z_candidates = model.criage_first_step(statements[subject_mask]).detach()

    entity_embedding = model.entity_representations[0](prediction[0]).detach()

    x_2 = z_candidates @ entity_embedding
    phi = torch.sigmoid(x_2)
    phi1mphi = phi * (1.0 - phi)

    outer = z_candidates.unsqueeze(2) * z_candidates.unsqueeze(1)

    m = hessian_head.unsqueeze(0) + phi1mphi.view(-1, 1, 1) * outer
    sol = torch.linalg.solve(m, z_candidates.unsqueeze(-1)).squeeze(-1)
    dot = (1 - phi).unsqueeze(-1) * sol

    relevance_head = (z_pred * dot).sum(dim=-1)

    if relevance_head.dtype  == torch.complex64 or relevance_head.dtype == torch.complex128:
        relevance_head = torch.abs(relevance_head)

    z_pred = model.criage_first_step(criage_prediction[object_mask]).detach()
    z_candidates = model.criage_first_step(statements[object_mask]).detach()

    entity_embedding = model.entity_representations[0](prediction[2]).detach()

    x_2 = z_candidates @ entity_embedding
    phi = torch.sigmoid(x_2)
    phi1mphi = phi * (1.0 - phi)

    outer = z_candidates.unsqueeze(2) * z_candidates.unsqueeze(1)

    m = hessian_tail.unsqueeze(0) + phi1mphi.view(-1, 1, 1) * outer
    sol = torch.linalg.solve(m, z_candidates.unsqueeze(-1)).squeeze(-1)
    dot = (1 - phi).unsqueeze(-1) * sol

    relevance_tail = (z_pred * dot).sum(dim=-1)

    if relevance_tail.dtype  == torch.complex64 or relevance_tail.dtype == torch.complex128:
        relevance_tail = torch.abs(relevance_tail)

    relevance = torch.zeros(n_statements).cuda()
    relevance[subject_mask] = relevance_head
    relevance[object_mask] = relevance_tail

    return -relevance
