import numpy as np
import torch
import sys
import torch.nn as nn
import csv
import os
import json

from torch.optim import Adam
from torch.utils.data import DataLoader

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

import models.helpers as helpers
import models.dataset_preparation as dataset_preparation

from models.baseline.baseline import Baseline2Layer

from models.baseline.baseline import train as train_baseline
from models.baseline.baseline import evaluate as evaluate_baseline
from models.use_st2.baseline_with_st2 import Baseline2LayerWST2
from models.use_st2.baseline_with_st2 import train as train_st2
from models.use_st2.baseline_with_st2 import evaluate as evaluate_st2
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    Baseline2LayerWST2LateFuse,
)

from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    train as train_later_fuse,
)
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    evaluate as evaluate_later_fuse,
)

models_dict = {
    "baseline2layer": (Baseline2Layer, train_baseline, evaluate_baseline),
    "late_fuse2layer": (
        Baseline2LayerWST2LateFuse,
        train_later_fuse,
        evaluate_later_fuse,
    ),
}

validated_hyperparams = {
    "baseline2layer": {
        "drop": 0.6,
        "threshold": 0.2,
        "lr": 1e-4,
        "batch_size": 16,
    },
    "late_fuse2layer": {
        "drop": 0.2,
        "threshold": 0.2,
        "lr": 1e-4,
        "batch_size": 16,
    },
}

if not os.path.exists("performance_at_epoch.json"):
    performance_at_epoch = {
        "baseline2layer": {},
        "late_fuse2layer": {},
    }
else:
    with open("performance_at_epoch.json") as f:
        performance_at_epoch = json.load(f)


def run_train_eval(
    model_name: str,
    drop: float = 0.5,
    threshold: float = 0.2,
    seed: int = 42,
    lr=1e-4,
    batch_size=16,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_datataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "train"
    )
    val_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "validation"
    )

    train_loader = DataLoader(
        train_datataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model, train, evaluate = models_dict[model_name]
    if "baseline" in model_name:
        model = model(
            train_datataset.embeddings_dimension,
            len(train_datataset.labels_transformer.classes),
            drop=drop,
        )
    else:
        model = model(
            train_datataset.embeddings_dimension,
            len(train_datataset.labels_transformer.classes),
            len(train_datataset.frames_transformer.classes),
            drop=drop,
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    patience = 5
    best_micro = 0
    macro_at_best_micro = 0
    best_epoch = 0

    for epoch in range(int(1e3)):
        train(model, optimizer, criterion, train_loader)
        macro_f1, micro_f1 = evaluate(
            model, criterion, val_loader, threshold=threshold
        )

        performance_at_epoch[model_name][str(epoch)] = performance_at_epoch[
            model_name
        ].get(str(epoch), []) + [micro_f1]

        print(f"Epoch {epoch + 1}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        if micro_f1 > best_micro:
            best_micro = micro_f1
            macro_at_best_micro = macro_f1
            best_epoch = epoch
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            break
    return (
        best_micro,
        macro_at_best_micro,
        best_epoch,
    )


if __name__ == "__main__":
    all_results = []
    seen_configs = set()
    RESULTS_CSV_FILENAME = "{model}_experiment_results_3.csv"

    seeds = [
        42,
        52,
        65,
        78,
        87,
        91,
        105,
        111,
        125,
        134,
        148,
        153,
        169,
        171,
        183,
        199,
        207,
        212,
        228,
        235,
        241,
        259,
        265,
        278,
        283,
        295,
        306,
        315,
        322,
        334,
    ]
    print(f"There are {len(seeds)} seeds to run")
    for seed in seeds:
        for model_name in models_dict.keys():
            if not os.path.exists(
                RESULTS_CSV_FILENAME.format(model=model_name)
            ):
                with open(
                    RESULTS_CSV_FILENAME.format(model=model_name),
                    "w",
                    newline="",
                ) as f:
                    f.write("\n")
            with open(RESULTS_CSV_FILENAME.format(model=model_name), "r") as f:
                reader = csv.DictReader(f)
                all_results = list(reader)

            for row in all_results:
                seen_configs.add(
                    (
                        row["model_name"],
                        row["seed"],
                        row["drop"],
                    )
                )

            threshold = validated_hyperparams[model_name]["threshold"]
            lr = validated_hyperparams[model_name]["lr"]
            batch_size = validated_hyperparams[model_name]["batch_size"]

            for drop in [0.2, 0.6]:
                if (
                    model_name,
                    str(seed),
                    str(drop),
                ) in seen_configs:
                    continue
                print(f"Running {model_name} with seed {seed}")
                (
                    best_micro,
                    macro_at_best_micro,
                    best_epoch,
                ) = run_train_eval(
                    model_name=model_name,
                    drop=drop,
                    threshold=threshold,
                    seed=seed,
                    lr=lr,
                    batch_size=batch_size,
                )
                result = {
                    "model_name": model_name,
                    "drop": drop,
                    "best_micro": best_micro,
                    "macro_at_best_micro": macro_at_best_micro,
                    "best_epoch": best_epoch,
                    "seed": seed,
                }

                all_results.append(result)

                with open(
                    RESULTS_CSV_FILENAME.format(model=model_name),
                    "w",
                    newline="",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(all_results[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(all_results)

                with open(f"performance_at_epoch.json", "w") as f:
                    json.dump(performance_at_epoch, f, indent=4)
