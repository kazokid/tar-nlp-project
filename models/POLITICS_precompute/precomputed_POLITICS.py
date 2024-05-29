import pandas as pd
from tqdm import tqdm
import sys
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
from bundle.scorers.scorer_subtask_3 import *
import bundle.baselines.st3 as bundle_baseline
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function
from models.dataset_preparation import *

import pickle


def make_dataframe_template(path: str, labels_fn: str = None):
    text = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        lines = file.read().splitlines()
        for line in lines:
            split_line = line.split("\t")
            iD = split_line[0]
            line_number = split_line[1]
            line_text = split_line[2]
            text.append((iD, line_number, line_text))

    df = pd.DataFrame(text, columns=["id", "line", "text"])
    print("DF loaded from path: " + path)
    print(df)
    df.id = df.id.apply(int)
    df.line = df.line.apply(int)
    df = df[df.text.str.strip().str.len() > 0].copy()
    df = df.set_index(["id", "line"])

    if labels_fn:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn, sep="\t", encoding="utf-8", header=None)
        labels = labels.rename(columns={0: "id", 1: "line", 2: "labels"})
        labels = labels.set_index(["id", "line"])
        # labels = labels[labels.labels.notna()].copy()
        labels["labels"] = labels["labels"].fillna("")
        # JOIN
        df = labels.merge(df, left_index=True, right_index=True)[
            ["text", "labels"]
        ]
    return df


def main():
    paths: dict = get_paths("en", "precomputed_POLITICS")

    translated_dev = paths["dev_template_translated"]
    out_fn = paths["dev_predictions"]

    # Read Data
    print("Loading dataset...")
    test = make_dataframe_template(translated_dev)

    with open("all_train_cased_labels.pkl", "rb") as f:
        Y_train = pickle.load(f)

    X_test = test["text"].values

    with open(CLASSES_SUBTASK_3_PATH, "r") as f:
        classes = f.read().split("\n")
        classes = [c for c in classes if c != ""]

    multibin = MultiLabelBinarizer(classes=classes)  # use sklearn binarizer

    Y_train = multibin.fit_transform(Y_train)
    # Create train-test split

    # load train embeddings
    with open("all_train_cased_embeddings.pkl", "rb") as f:
        X_train = pickle.load(f)

    # load test embeddings
    with open("embeddings_en_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    if len(X_train) != len(Y_train):
        raise ValueError("X_train and Y_train must have the same length")

    print("Fitting logistic regression...")
    model = MultiOutputClassifier(
        LogisticRegression(tol=1e-4, max_iter=int(1e7), n_jobs=-1, verbose=1)
    )
    model.fit(X_train, Y_train)

    print("In-sample Acc: \t\t", model.score(X_train, Y_train))

    Y_pred = model.predict(X_test)
    out = multibin.inverse_transform(Y_pred)
    out = list(map(lambda x: ",".join(x), out))
    out = pd.DataFrame(out, test.index)
    out.to_csv(out_fn, sep="\t", header=None)
    print("Results on: ", out_fn)


if __name__ == "__main__":
    main()
