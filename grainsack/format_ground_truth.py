"""Kelpie model for PyKEEN."""

import numpy as np
import pandas as pd
from pykeen.triples import TriplesFactory

from grainsack import KGS_PATH

DATASET = "FR200K"


def format_fr200k():
    """Format the FR200K ground-dataset.
    
    Convert it to a list of triples.
    Split the triples into training, validation, and testing triples.
    Create a .csv file associating each triple to its explanations.
    """
    data = np.load(KGS_PATH / "FR200K" / f"{DATASET}.npz")

    triples = data["all_triples"]
    xsets = data["all_traces"]
    wsets = data["all_weights"]

    dbpedia_url_base = "http://dbpedia.org/resource/"

    triples = triples.tolist()
    triples = [(s.replace(dbpedia_url_base, ""), p, o.replace(dbpedia_url_base, "")) for s, p, o in triples]
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
    df["explanation"] = df["explanation"].map(
        lambda x: [[s.replace(dbpedia_url_base, "")[1:-1], p, o.replace(dbpedia_url_base, "")[1:-1]] for s, p, o in x]
    )
    df["weight"] = df["weight"].astype(float)
    df["weight"] = pd.qcut(df["weight"], 3, labels=[-1, 0, 1])

    df.to_csv(KGS_PATH / "FR200K" / "explanations.txt", sep="\t", index=False, header=False)

    tf = TriplesFactory.from_path("explanations.txt")

    training, testing, validation = tf.split([0.8, 0.1, 0.1])

    train_df = training.tensor_to_df(training.mapped_triples)
    train_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    train_df.to_csv(KGS_PATH / "FR200K" / "train.txt", sep="\t", index=False, header=False)

    test_df = testing.tensor_to_df(testing.mapped_triples)
    test_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    test_df.to_csv(KGS_PATH / "FR200K" / "test.txt", sep="\t", index=False, header=False)

    valid_df = validation.tensor_to_df(validation.mapped_triples)
    valid_df.drop(columns=["head_id", "relation_id", "tail_id"], inplace=True)
    valid_df.to_csv(KGS_PATH / "FR200K" / "valid.txt", sep="\t", index=False, header=False)


def format_fruni():
    """Format the FRUNI dataset for training and testing."""
    input_file = KGS_PATH / "FRUNI" / "test_explanations.txt"

    explained_predictions = []
    with open(input_file, "r", encoding="") as infile:
        for line in infile:
            values = line.strip().split(",")
            if len(values) > 3:
                explained_predictions.append(values)

    explained_predictions = [
        {
            "pred": (ep[0], ep[1], ep[2]),
            "explanation": [(ep[i], ep[i + 1], ep[i + 2]) for i in range(3, len(ep), 3)],
            "label": "1",
        }
        for ep in explained_predictions
    ]

    df = pd.DataFrame(explained_predictions)
    df.to_csv(KGS_PATH / "FRUNI" / "explanations.txt", sep="\t", index=False, header=False)


if __name__ == "__main__":
    format_fr200k()
