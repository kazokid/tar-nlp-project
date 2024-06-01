import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn
import csv

from sklearn.metrics import classification_report as accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.optim import Adam
from torch.utils.data import DataLoader

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
import bundle.scorers.scorer_subtask_3 as bundle_scorer
import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers
import models.dataset_preparation as dataset_preparation

from models.baseline.baseline import (
    Baseline3Layer,
    Baseline4Layer,
    Baseline5Layer,
)
from models.baseline.baseline import train as train_baseline
from models.baseline.baseline import evaluate as evaluate_baseline
from models.use_st2.baseline_with_st2 import (
    Baseline3LayerWST2,
    Baseline4LayerWST2,
    Baseline5LayerWST2,
)
from models.use_st2.baseline_with_st2 import train as train_st2
from models.use_st2.baseline_with_st2 import evaluate as evaluate_st2
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    Baseline3LayerWST2LateFuse,
    Baseline4LayerWST2LateFuse,
    Baseline5LayerWST2LateFuse,
)
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    train as train_later_fuse,
)
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    evaluate as evaluate_later_fuse,
)

models_dict = {
    "baseline3layer": (Baseline3Layer, train_baseline, evaluate_baseline),
    "baseline4layer": (Baseline4Layer, train_baseline, evaluate_baseline),
    "baseline5layer": (Baseline5Layer, train_baseline, evaluate_baseline),
    "use_st2_3layer": (Baseline3LayerWST2, train_st2, evaluate_st2),
    "use_st2_4layer": (Baseline4LayerWST2, train_st2, evaluate_st2),
    "use_st2_5layer": (Baseline5LayerWST2, train_st2, evaluate_st2),
    "use_st2_late_fuse_3layer": (
        Baseline3LayerWST2LateFuse,
        train_later_fuse,
        evaluate_later_fuse,
    ),
    "use_st2_late_fuse_4layer": (
        Baseline4LayerWST2LateFuse,
        train_later_fuse,
        evaluate_later_fuse,
    ),
    "use_st2_late_fuse_5layer": (
        Baseline5LayerWST2LateFuse,
        train_later_fuse,
        evaluate_later_fuse,
    ),
}


def main(model_name: str, drop: float = 0.5, threshold: float = 0.2) -> None:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_datataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "train"
    )
    val_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "dev"
    )

    val_eng_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", ["en"], "dev"
    )
    val_it_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", ["it"], "dev"
    )

    train_loader = DataLoader(train_datataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_eng_loader = DataLoader(val_eng_dataset, batch_size=32, shuffle=False)
    val_it_loader = DataLoader(val_it_dataset, batch_size=32, shuffle=False)
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
    optimizer = Adam(model.parameters(), lr=1e-4)

    patience = 5
    best_micro = 0
    macro_at_best_micro = 0
    best_epoch = 0
    micro_f1_eng_at_best_micro = 0
    micro_f1_it_at_best_micro = 0
    for epoch in range(int(1e3)):
        train(model, optimizer, criterion, train_loader)
        macro_f1, micro_f1 = evaluate(
            model, criterion, val_loader, threshold=threshold
        )
        macro_f1_eng, micro_f1_eng = evaluate(
            model, criterion, val_eng_loader, threshold=threshold
        )
        macro_f1_it, micro_f1_it = evaluate(
            model, criterion, val_it_loader, threshold=threshold
        )

        print(f"Epoch {epoch + 1}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        if micro_f1 > best_micro:
            best_micro = micro_f1
            macro_at_best_micro = macro_f1
            best_epoch = epoch
            micro_f1_eng_at_best_micro = micro_f1_eng
            micro_f1_it_at_best_micro = micro_f1_it
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            break
    return (
        best_micro,
        macro_at_best_micro,
        best_epoch,
        micro_f1_eng_at_best_micro,
        micro_f1_it_at_best_micro,
    )


if __name__ == "__main__":
    all_results = []
    seen_configs = set()
    RESULTS_CSV_FILENAME = "hyperparam_results_2.csv"
    with open(RESULTS_CSV_FILENAME, "r") as f:
        reader = csv.DictReader(f)
        all_results = list(reader)

    for row in all_results:
        seen_configs.add((row["model_name"], row["drop"], row["threshold"]))

    for model_name in models_dict.keys():
        for drop in [0.3, 0.4, 0.5, 0.6]:
            for threshold in [0.2]:
                if (model_name, str(drop), str(threshold)) in seen_configs:
                    continue
                print(
                    f"Running {model_name} with drop {drop} and threshold {threshold}"
                )
                (
                    best_micro,
                    macro_at_best_micro,
                    best_epoch,
                    micro_f1_eng_at_best_micro,
                    micro_f1_it_at_best_micro,
                ) = main(model_name, drop, threshold)
                all_results.append(
                    {
                        "model_name": model_name,
                        "drop": drop,
                        "threshold": threshold,
                        "best_micro": best_micro,
                        "macro_at_best_micro": macro_at_best_micro,
                        "best_epoch": best_epoch,
                        "mi_f1_eng_at_best_micro": micro_f1_eng_at_best_micro,
                        "mi_f1_it_at_best_micro": micro_f1_it_at_best_micro,
                    }
                )
                with open(RESULTS_CSV_FILENAME, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "model_name",
                            "drop",
                            "threshold",
                            "best_micro",
                            "macro_at_best_micro",
                            "best_epoch",
                            "mi_f1_eng_at_best_micro",
                            "mi_f1_it_at_best_micro",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(all_results)
