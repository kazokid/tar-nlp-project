import numpy as np
import torch
import sys
import torch.nn as nn
import csv
import os

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
    "early_fuse2layer": (Baseline2LayerWST2, train_st2, evaluate_st2),
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
    "early_fuse2layer": {
        "drop": 0.4,
        "threshold": 0.2,
        "lr": 1e-4,
        "batch_size": 32,
    },
    "late_fuse2layer": {
        "drop": 0.2,
        "threshold": 0.2,
        "lr": 1e-4,
        "batch_size": 16,
    },
}

saved_models = {
    "baseline2layer": {
        "path": "baseline2layer.pth",
        "best_validation_micro": 0.0,
    },
    "early_fuse2layer": {
        "path": "early_fuse2layer.pth",
        "best_validation_micro": 0.0,
        "seed": 0,
    },
    "late_fuse2layer": {
        "path": "late_fuse2layer.pth",
        "best_validation_micro": 0.0,
        "seed": 0,
    },
}


def train_test_eval(
    model_name: str,
    drop: float = 0.5,
    threshold: float = 0.2,
    seed: int = 42,
    lr=1e-4,
    batch_size=16,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_per_lang = {}
    for lang in helpers.languages_train:
        test_per_lang[lang] = {
            "dataset": dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
                "bert-base-multilingual-cased", [lang], "test"
            ),
            "micro_f1": 0,
            "macro_f1": 0,
            "best_micro_f1": 0,
            "best_macro_f1": 0,
        }
        test_per_lang[lang]["loader"] = DataLoader(
            test_per_lang[lang]["dataset"], batch_size=32, shuffle=False
        )

    train_datataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "train"
    )
    val_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "validation"
    )
    test_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "test"
    )

    train_loader = DataLoader(
        train_datataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    test_micro_best = 0
    test_macro_best = 0
    best_epoch = 0

    for epoch in range(int(1e3)):
        train(model, optimizer, criterion, train_loader)
        macro_f1, micro_f1 = evaluate(
            model, criterion, val_loader, threshold=threshold
        )
        test_macro_f1, test_micro_f1 = evaluate(
            model, criterion, test_loader, threshold=threshold
        )

        for lang in helpers.languages_train:
            test_per_lang[lang]["macro_f1"], test_per_lang[lang]["micro_f1"] = (
                evaluate(
                    model,
                    criterion,
                    test_per_lang[lang]["loader"],
                    threshold=threshold,
                )
            )

        print(f"Epoch {epoch + 1}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        if micro_f1 > best_micro:
            best_micro = micro_f1
            macro_at_best_micro = macro_f1
            test_micro_best = test_micro_f1
            test_macro_best = test_macro_f1
            if best_micro > saved_models[model_name]["best_validation_micro"]:
                torch.save(model.state_dict(), saved_models[model_name]["path"])
                saved_models[model_name]["best_validation_micro"] = best_micro
                saved_models[model_name]["seed"] = seed
            best_epoch = epoch

            for lang in helpers.languages_train:
                test_per_lang[lang]["best_micro_f1"] = test_per_lang[lang][
                    "micro_f1"
                ]
                test_per_lang[lang]["best_macro_f1"] = test_per_lang[lang][
                    "macro_f1"
                ]
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            break
    return (
        best_micro,
        macro_at_best_micro,
        best_epoch,
        test_micro_best,
        test_macro_best,
        test_per_lang,
    )


if __name__ == "__main__":
    all_results = []
    seen_configs = set()
    RESULTS_CSV_FILENAME = "{model}_test_results_1.csv"

    for model_name in models_dict.keys():
        if not os.path.exists(RESULTS_CSV_FILENAME.format(model=model_name)):
            with open(
                RESULTS_CSV_FILENAME.format(model=model_name), "w", newline=""
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
                )
            )
            if (
                float(row["best_micro"])
                > saved_models[model_name]["best_validation_micro"]
            ):
                saved_models[model_name]["best_validation_micro"] = float(
                    row["best_micro"]
                )
                saved_models[model_name]["seed"] = int(row["seed"])

        drop = validated_hyperparams[model_name]["drop"]
        threshold = validated_hyperparams[model_name]["threshold"]
        lr = validated_hyperparams[model_name]["lr"]
        batch_size = validated_hyperparams[model_name]["batch_size"]

        seeds = [42, 65, 456, 789, 100]
        for seed in seeds:
            if (
                model_name,
                str(seed),
            ) in seen_configs:
                continue
            print(f"Running {model_name} with seed {seed}")
            (
                best_micro,
                macro_at_best_micro,
                best_epoch,
                test_micro_best,
                test_macro_best,
                test_per_lang,
            ) = train_test_eval(
                model_name=model_name,
                drop=drop,
                threshold=threshold,
                seed=seed,
                lr=lr,
                batch_size=batch_size,
            )
            result = {
                "model_name": model_name,
                "best_micro": best_micro,
                "macro_at_best_micro": macro_at_best_micro,
                "best_epoch": best_epoch,
                "seed": seed,
                "test_micro_best": test_micro_best,
                "test_macro_best": test_macro_best,
            }

            for lang in helpers.languages_train:
                result[f"{lang}_best_micro"] = test_per_lang[lang][
                    "best_micro_f1"
                ]
                result[f"{lang}_best_macro"] = test_per_lang[lang][
                    "best_macro_f1"
                ]

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

        print(
            f"Finished {model_name}, best seed {saved_models[model_name]['seed']}"
        )
