import pandas as pd
from tqdm import tqdm
import os
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
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


class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, model, max_length=512):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoding = [
            self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            for text in X
        ]
        with torch.no_grad():
            return [
                self.model(
                    input_ids=e["input_ids"], attention_mask=e["attention_mask"]
                )[0][0][0].numpy()
                for e in encoding
            ]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

pipeline = Pipeline(
    [
        ("bert_vectorizer", BertVectorizer(tokenizer, model)),
        ("multi_output", MultiOutputClassifier(LogisticRegression())),
    ]
)


def main():
    paths: dict = get_paths("en", "pipeline_bert")

    folder_train = paths["train_folder"]
    folder_dev = paths["dev_folder"]
    labels_train_fn = paths["train_labels"]
    out_fn = paths["dev_predictions"]

    # Read Data
    print("Loading dataset...")
    train = bundle_baseline.make_dataframe(folder_train, labels_train_fn)
    test = bundle_baseline.make_dataframe(folder_dev)

    X_train = train["text"].values
    Y_train = train["labels"].fillna("").str.split(",").values

    X_test = test["text"].values

    multibin = MultiLabelBinarizer()  # use sklearn binarizer

    Y_train = multibin.fit_transform(Y_train)
    # Create train-test split

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    pipe = Pipeline(
        [
            ("bert_vectorizer", BertVectorizer(tokenizer, model)),
            (
                "multi_output",
                MultiOutputClassifier(
                    svm.SVC(tol=1e-2, verbose=True, max_iter=1)
                ),
            ),
        ],
        memory="C:/tmp/tar_pipeline_cache",
    )

    print("Fitting logistic regression...")  # svm
    pipe.fit(X_train, Y_train)

    print("In-sample Acc: \t\t", pipe.score(X_train, Y_train))

    Y_pred = pipe.predict(X_test)
    out = multibin.inverse_transform(Y_pred)
    out = list(map(lambda x: ",".join(x), out))
    out = pd.DataFrame(out, test.index)
    out.to_csv(out_fn, sep="\t", header=None)
    print("Results on: ", out_fn)


if __name__ == "__main__":
    main()
