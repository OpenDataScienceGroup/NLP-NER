import json
from itertools import combinations
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

O = 8

tags = {
    "PER": {3, 7},
    "LOC": {0, 4},
    "ORG": {2, 6},
    "MISC": {1, 5},
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def filter_entity(tags, allowed):
    return [
        tag if tag in allowed else O
        for tag in tags
    ]


def entity_f1(tags_a, tags_b):
    #print(tags_a,tags_b)
    y_true = []
    y_pred = []

    for a, b in zip(tags_a, tags_b):
        if a == O and b == O:
            continue

        y_true.append(a)
        y_pred.append(b)

    if not y_true:
        return 1.0

    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    return f1


def iaa(files:dict):

    results = {}

    for entity, allowed_tags in tags.items():

        pair_scores = {}

        for a, b in combinations(files.keys(),2):
            #print(a,b)
            scores = []

            for ex_a, ex_b in zip(files[a], files[b]):

                assert ex_a["tokens"] == ex_b["tokens"]

                tags_a = filter_entity(
                    ex_a["tags"],
                    allowed_tags
                )

                tags_b = filter_entity(
                    ex_b["tags"],
                    allowed_tags
                )

                scores.append(
                    entity_f1(tags_a, tags_b)
                )
            pair_scores[f"{a}-{b}"] = np.mean(scores)
        results[entity] = {
            **pair_scores,
            "mean": np.mean(list(pair_scores.values()))
        }

    return results


files = {
    "1":load_jsonl("../data/Hjalte_500.jsonl"),
    "2":load_jsonl("../data/mar_500.jsonl"),
    "3":load_jsonl("../data/val_500.jsonl")#,
    #"Gold":load_jsonl("../data/test.jsonl"),
    #"bertweet":load_jsonl("../data/bertweet_pred.jsonl")
}

results = iaa(files)

for entity, scores in results.items():
    print(f"\n{entity}")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")
