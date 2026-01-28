import heapq
import itertools
import os
import subprocess
import uuid
from functools import partial
from time import time

import torch
from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from tqdm import tqdm

from grainsack import logger
from grainsack.kge_lp import complex_cosine_similarity


SHM = "/dev/shm"


def mhs_explain_factory(kg, kge_model, kge_config, lpx_config):
    return partial(mhs_explain, kg, kge_model, k=2)


def negate_assertion(s, p, o):
    if p == RDF.type:
        not_c = BNode()
        return {
            (not_c, RDF.type, OWL.Class),
            (not_c, OWL.complementOf, o),
            (s, RDF.type, not_c),
        }
    if p != RDF.type:
        neg = BNode()
        return {
            (neg, RDF.type, OWL.NegativePropertyAssertion),
            (neg, OWL.sourceIndividual, s),
            (neg, OWL.assertionProperty, p),
            (neg, OWL.targetIndividual, o),
        }


def local_name(uri):
    if not isinstance(uri, URIRef):
        return str(uri)
    return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]


def pretty_dl(triples):
    out = []

    if triples is None:
        return "None"

    for s, p, o in triples:
        s_n = local_name(s)
        p_n = local_name(p)
        o_n = local_name(o)

        if p == RDF.type:
            out.append(f"{s_n} : {o_n}")
        else:
            out.append(f"({s_n}, {o_n}) : {p_n}")

    return out


def consistent(kg, addition):
    start_serialization = time()
    temp_file_name = f"konclude_{uuid.uuid4().hex}"
    temp_file_xml = f"{SHM}/{temp_file_name}.xml"

    kgp = Graph()
    kgp.addN((s, p, o, kgp) for (s, p, o) in kg)
    kgp.addN((s, p, o, kgp) for (s, p, o) in addition)

    kgp.serialize(destination=temp_file_xml, format="xml")

    end_serialization = time()

    start_reasoning = time()
    result = subprocess.run(
        ["./Konclude/Binaries/Konclude", "consistency", "-w", "AUTO", "-i", temp_file_xml], capture_output=True, text=True
    )
    output = result.stdout
    end_reasoning = time()

    logger.info(
        f"Serialization: {end_serialization - start_serialization:.2f}s, Reasoning: {end_reasoning - start_reasoning:.2f}s")

    os.remove(temp_file_xml)

    return "is consistent" in output.lower()


def get_abducibles(kg, prediction, kge_model, k):
    abd_inds = set()
    abd_preds = set()

    classes = set(kg.rdf_kg.subjects(RDF.type, OWL.Class)) | set(
        kg.rdf_kg.subjects(RDF.type, RDFS.Class))
    classes_id = {int(kg.entity_to_id[str(c)])
                  for c in classes if str(c) in kg.entity_to_id}

    obj_props = set(kg.rdf_kg.subjects(RDF.type, OWL.ObjectProperty))
    obj_props_id = {int(kg.relation_to_id[str(p)])
                    for p in obj_props if str(p) in kg.relation_to_id}

    inds = set(kg.rdf_kg.subjects(RDF.type, OWL.NamedIndividual))
    inds_id = {int(kg.entity_to_id[str(i)])
               for i in inds if str(i) in kg.entity_to_id}

    s_ = kge_model.entity_representations[0](torch.tensor(
        kg.entity_to_id[str(prediction[0])], dtype=torch.long).cuda()).detach()
    p_ = kge_model.relation_representations[0](
        torch.tensor(kg.relation_to_id[str(
            prediction[1])], dtype=torch.long).cuda()
    ).detach()
    o_ = kge_model.entity_representations[0](torch.tensor(
        kg.entity_to_id[str(prediction[2])], dtype=torch.long).cuda()).detach()

    inds_ = kge_model.entity_representations[0](
        torch.tensor(list(inds_id), dtype=torch.long).cuda()).detach()
    obj_props_ = kge_model.relation_representations[0](
        torch.tensor(list(obj_props_id), dtype=torch.long).cuda()).detach()
    classes_ = kge_model.entity_representations[0](
        torch.tensor(list(classes_id), dtype=torch.long).cuda()).detach()

    if inds_.dtype == torch.cfloat:
        s_sim_inds = complex_cosine_similarity(s_.unsqueeze(0), inds_, dim=1)
        o_sim_inds = complex_cosine_similarity(o_.unsqueeze(0), inds_, dim=1)
        s_sim_classes = complex_cosine_similarity(
            s_.unsqueeze(0), classes_, dim=1)
        o_sim_classes = complex_cosine_similarity(
            o_.unsqueeze(0), classes_, dim=1)
        p_sim = complex_cosine_similarity(p_.unsqueeze(0), obj_props_, dim=1)
    else:
        s_sim_inds = torch.nn.functional.cosine_similarity(
            s_.unsqueeze(0), inds_, dim=1)
        o_sim_inds = torch.nn.functional.cosine_similarity(
            o_.unsqueeze(0), inds_, dim=1)
        s_sim_classes = torch.nn.functional.cosine_similarity(
            s_.unsqueeze(0), classes_, dim=1)
        o_sim_classes = torch.nn.functional.cosine_similarity(
            o_.unsqueeze(0), classes_, dim=1)
        p_sim = torch.nn.functional.cosine_similarity(
            p_.unsqueeze(0), obj_props_, dim=1)

    sim_inds = (s_sim_inds + o_sim_inds) / 2.0
    sim_classes = (s_sim_classes + o_sim_classes) / 2.0

    _, topk_inds_indices = torch.topk(sim_inds, 5)
    _, topk_preds_indices = torch.topk(p_sim, 2)

    for idx in topk_inds_indices:
        abd_inds.add(URIRef(kg.id_to_entity[idx.item()]))

    abd_inds.add(URIRef(prediction[0]))

    class_assertions = set()
    for ind in abd_inds:
        added_ca = 0
        for idx in torch.argsort(sim_classes, descending=True):
            if added_ca > k:
                break
            class_id = list(classes_id)[idx.item()]
            class_uri = URIRef(kg.id_to_entity[class_id])
            if (ind, RDF.type, class_uri) != prediction and (ind, RDF.type, class_uri) not in kg.rdf_kg:
                class_assertions.add((ind, RDF.type, class_uri))
                added_ca += 1

    for idx in topk_preds_indices:
        abd_preds.add(URIRef(kg.id_to_relation[idx.item()]))

    objprop_assertions = set()
    for s in abd_inds:
        for o in abd_inds:
            if s == o:
                continue
            added_a = 0
            for idx in torch.argsort(p_sim, descending=True):
                if added_a > k:
                    break
                obj_prop_id = list(obj_props_id)[idx.item()]
                obj_prop_uri = URIRef(kg.id_to_relation[obj_prop_id])
                if (s, obj_prop_uri, o) != prediction and (s, obj_prop_uri, o) not in kg.rdf_kg:
                    objprop_assertions.add((s, obj_prop_uri, o))
                    added_a += 1

    return class_assertions | objprop_assertions


def find_min_conflict(r, a, k, u):
    if consistent(k, u):
        return {}
    return find_min_conflict_(r, a, k, u)


def find_min_conflict_(r, a, k, u):
    logger.info(f"find_min_conflict_: r={len(r)}, a={len(a)}")

    if r and consistent(k, u - r):
        return set()

    if len(a) == 1:
        return a

    a_list = list(a)

    mid = (len(a_list) + 1) // 2

    a1 = set(a_list[:mid])
    a2 = set(a_list[mid:])

    x1 = find_min_conflict_(r | a2, a1, k, u)
    x2 = find_min_conflict_(r | x1, a2, k, u)

    return x1.union(x2)


def get_solutions(kg, not_observation, abducibles, max_depth=2):
    pq = []

    def push_path(p):
        fs = frozenset(p)
        heapq.heappush(pq, (len(fs), next(counter), fs))

    counter = itertools.count()

    solutions = []
    conflicts = []

    generated = set()

    push_path(set())

    while pq:
        depth, _, path = heapq.heappop(pq)

        logger.info(
            f"[MHS] Exploring node {path} | depth={depth} | path_size={len(path)}")

        if depth > max_depth:
            break

        if path in generated:
            continue
        generated.add(path)

        if any(s <= path for s in solutions):
            continue

        if not consistent(kg | not_observation | path, set()):
            if consistent(kg | path, set()):
                return path
            solutions.append(path)
            continue

        conflict = None
        for c in conflicts:
            if path.isdisjoint(c):
                conflict = c
                break
        if conflict is None:
            logger.info("[MHS] Computing new conflict")
            conflict = find_min_conflict(
                set(), abducibles - path, kg | not_observation | path, abducibles)
            conflicts.append(conflict)
            logger.info(f"[MHS] New conflict of size {len(conflict)}")

        for e in conflict:
            push_path(path | {e})


def get_justification(kg, addition):
    temp_file_name = f"robot_{uuid.uuid4().hex}"
    temp_file_xml = f"{SHM}/{temp_file_name}.xml"

    kgp = Graph()
    kgp.addN((s, p, o, kgp) for (s, p, o) in kg)
    kgp.addN((s, p, o, kgp) for (s, p, o) in addition)

    kgp.serialize(destination=temp_file_xml, format="xml")

    inconsistency_file_name = f"inconsistency_{uuid.uuid4().hex}.md"

    subprocess.run(
        [
            "java",
            "-jar",
            "-Xms1g",
            "-Xmx128g",
            "/leonardo_work/IscrC_MINA/roberto/research/robot/robot.jar",
            "explain",
            "--input",
            temp_file_xml,
            "--reasoner",
            "jfact",
            "-M",
            "inconsistency",
            "-m",
            "1",
            "--explanation",
            inconsistency_file_name,
        ]
    )

    # read the content of inconsistency.md
    try:
        with open(inconsistency_file_name, "r", encoding="utf-8") as f:
            explanation = f.read()
        os.remove(inconsistency_file_name)
    except FileNotFoundError as e:
        explanation = "No justification found."
        logger.info(f"Justification file not found: {e}")

    os.remove(temp_file_xml)

    return explanation


def mhs_explain(kg, kge_model, prediction, k=5):
    prediction = (URIRef(prediction[0]), URIRef(
        prediction[1]), URIRef(prediction[2]))
    logger.info("Getting abducibles")
    abducibles = get_abducibles(kg, prediction, kge_model, k) - {prediction}
    logger.info(f"Number of abducibles: {len(abducibles)}")
    kg = set(kg.rdf_kg.triples((None, None, None)))
    not_observation = negate_assertion(*prediction)

    logger.info("Searching for solution")
    solution = get_solutions(kg, not_observation, abducibles)

    if solution is None:
        return [[]]

    logger.info(f"Solution found with {len(solution)} assertions")

    logger.info("Getting justification")
    justification = get_justification(kg, solution | not_observation)
    logger.info("Justification obtained")

    return [justification]
