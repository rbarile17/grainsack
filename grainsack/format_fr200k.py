import os

import numpy as np
import pandas as pd
from pykeen.triples import TriplesFactory


def concat_triples(data, rules):

    triples = []
    traces = []
    weights = []

    for rule in rules:

        triple_name = rule + "_triples"
        traces_name = rule + "_traces"
        weights_name = rule + "_weights"

        triples.append(data[triple_name])
        traces.append(data[traces_name])
        weights.append(data[weights_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights, axis=0)

    return triples, traces, weights


def get_data(data, rule):

    if rule == "full_data":

        triples = data["all_triples"]
        traces = data["all_traces"]
        weights = data["all_weights"]

        entities = data["all_entities"].tolist()
        relations = data["all_relations"].tolist()

    else:
        triples, traces, weights = concat_triples(data, [rule])
        entities = data[rule + "_entities"].tolist()
        relations = data[rule + "_relations"].tolist()

    return triples, traces, weights, entities, relations


DATASET = "FR200K"
RULE = "full_data"


def format_fr200k():
    data = np.load(os.path.join("kgs/FR200K/", DATASET + ".npz"))

    triples, xsets, wsets, entities, relations = get_data(data, RULE)

    dbpedia_url_base = "http://dbpedia.org/resource/"
    strip_url_base = lambda entity: entity.replace(dbpedia_url_base, "")

    triples = triples.tolist()
    triples = [(strip_url_base(s), p, strip_url_base(o)) for s, p, o in triples]
    triples = [(s[1:-1], p, o[1:-1]) for s, p, o in triples]

    xsets = [xset.tolist() for xset in xsets]
    xsets = [
        {
            "explanations": [
                {"triples": [t for t in x if t[0] != "UNK_ENT"], "weight": w[0]}
                for x, w in zip(xset, wset)
                if x[0][0] != "UNK_ENT"
            ]
        }
        for xset, wset in zip(xsets, wsets)
    ]

    triples = [(s, p, o, xset) for (s, p, o), xset in zip(triples, xsets) if xset["explanations"]]
    triples = [(s, p, o, e["triples"], e["weight"]) for s, p, o, xset in triples for e in xset["explanations"]]
    triples = [(s, p, o, e, w) for s, p, o, e, w in triples if e]

    df = pd.DataFrame(triples, columns=["s", "p", "o", "explanation", "weight"])
    df["explanation"] = df["explanation"].map(lambda x: [[strip_url_base(s)[1:-1], p, strip_url_base(o)[1:-1]] for s, p, o in x])
    df["weight"] = df["weight"].astype(float)
    df["weight"] = pd.qcut(df["weight"], 3, labels=[-1, 0, 1])

    # triples = df.to_dict(orient="records")

    # df = pd.DataFrame(triples)
    df.to_csv("data.txt", sep="\t", index=False, header=False)

    tf = TriplesFactory.from_path("data.txt")

    training, testing, validation = tf.split([0.8, 0.1, 0.1])

    train_df = training.tensor_to_df(training.mapped_triples)
    train_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    train_df.to_csv("kgs/FR200K/train.txt", sep="\t", index=False, header=False)

    test_df = testing.tensor_to_df(testing.mapped_triples)
    test_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    test_df.to_csv("kgs/FR200K/test.txt", sep="\t", index=False, header=False)

    valid_df = validation.tensor_to_df(validation.mapped_triples)
    valid_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    valid_df.to_csv("kgs/FR200K/valid.txt", sep="\t", index=False, header=False)

if __name__ == "__main__":
    format_fr200k()
