import pandas as pd
from tqdm import tqdm
import os
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from sklearn.linear_model import LogisticRegression

from transformers import BertTokenizer, BertModel

from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
from bundle.scorers.scorer_subtask_3 import *
import bundle.baselines.st3 as bundle_baseline
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function
from models.dataset_preparation import *

import pickle


def main():
    paths: dict = get_paths("it", "precomputed_all_bert")

    folder_dev = paths["dev_folder"]
    out_fn = paths["dev_predictions"]

    # Read Data
    print("Loading dataset...")
    test = bundle_baseline.make_dataframe(folder_dev)

    with open("all_train_cased_labels.pkl", "rb") as f:
        Y_train = pickle.load(f)

    X_test = test["text"].values

    multibin = MultiLabelBinarizer()  # use sklearn binarizer

    Y_train = multibin.fit_transform(Y_train)
    # Create train-test split

    # load train embeddings
    with open("all_train_cased_embeddings.pkl", "rb") as f:
        X_train = pickle.load(f)

    # load test embeddings
    with open("embeddings_it_cased_test.pkl", "rb") as f:
        X_test = pickle.load(f)

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
