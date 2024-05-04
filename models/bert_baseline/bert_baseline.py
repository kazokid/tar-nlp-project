import numpy as np
import pandas as pd
import torch
import sys
import argparse
import os
import torch.nn as nn

from sklearn.metrics import classification_report as report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from torch.optim import Adam

# TODO check if this works
if os.getenv("MY_REPO_LOCATION") not in sys.path:
    sys.path.append(
        os.getenv("MY_REPO_LOCATION")
    )  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file, evaluate
from bundle.baselines.st3 import *
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function


class BertBaseline(nn.Module):
    def __init__(self, language: str) -> None:
        super().__init__()
        self.language = language
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.bert_model = BertModel.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.loss_function = (
            torch.nn.BCEWithLogitsLoss()
        )  # check if this is the correct loss function

        self.linear1 = torch.nn.Linear(BertModel.config.hidden_size, 100)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(100, 20)  # 20 classes??
        # self.sigmoid = torch.nn.Sigmoid() # this is not needed since BCEWithLogitsLoss already applies sigmoid

    def added_parameters(self):
        """Returns the parameters of the model that need to be optimized - excluding the BERT model parameters"""
        return list(self.linear1.parameters()) + list(self.linear2.parameters())

    def forward(self, x):
        tokenized_input = self.bert_tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        )  # check if max_length is correct

        outputs = self.bert_model(**tokenized_input)

        last_hidden_states = outputs.last_hidden_state
        linear_output = self.linear1(last_hidden_states)
        activation_output = self.activation(linear_output)
        classification_output = self.linear2(activation_output)
        # sigmoid_output = self.sigmoid(classification_output) # this is not needed since BCEWithLogitsLoss already applies sigmoid

        return classification_output

    def train(self, X_train, Y_train, batch_size=16, n_epochs=5, lr=2e-5):
        optimizer = Adam(self.added_parameters(), lr=lr)
        multibin = MultiLabelBinarizer()
        Y_train = multibin.fit_transform(Y_train)

        for epoch in range(n_epochs):
            total_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                self.train()  # set model to training mode, possibly useless (no dropout or batchnorm layers in this model)

                batch_texts = X_train[i : i + batch_size]
                Y_batch = Y_train[i : i + batch_size]

                clasification_measures = self.forward(batch_texts)
                loss = self.loss_function(
                    clasification_measures,
                    torch.tensor(Y_batch, dtype=torch.float32),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / (len(X_train) // batch_size)
            print("Epoch:", epoch + 1, "Average Batch Loss:", average_loss)

        return average_loss

    def predict(self, X, threshold=0.5):
        self.eval()  # set model to evaluation mode, possibly useless (no dropout or batchnorm layers in this model)

        with torch.no_grad():
            clasification_measures = self.forward(X)
            sigmoid_output = torch.sigmoid(clasification_measures)
            sigmoid_output[sigmoid_output >= threshold] = 1
            sigmoid_output[sigmoid_output < threshold] = 0

        return sigmoid_output


def second_main():
    # TODO check and finish to run the class model
    parser = argparse.ArgumentParser(description="Subtask-3")
    parser.add_argument(
        "language",
        type=str,
        nargs=1,
        help="Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)",
    )

    args = parser.parse_args()
    language = args.language[0]
    paths: dict = get_paths(language)

    out_fn = os.path.join(
        BASE_PATH, f"outputs/bert_baseline/{language}_output.txt"
    )
    out_true = os.path.join(
        BASE_PATH, f"outputs/bert_baseline/{language}_output_true.txt"
    )

    out_dev = os.path.join(
        BASE_PATH, f"outputs/bert_baseline/{language}_dev_output.txt"
    )
    out_dev_true = os.path.join(
        BASE_PATH, f"outputs/bert_baseline/{language}_dev_output_true.txt"
    )

    classes = (
        CLASSES_SUBTASK_3_PATH  # this never changes so we can use the constant
    )

    labels = pd.read_csv(
        paths["train_labels"], sep="\t", encoding="utf-8", header=None
    )
    labels = labels.rename(columns={0: "id", 1: "line", 2: "labels"})
    labels = labels.set_index(["id", "line"])

    labels_dev = pd.read_csv(
        paths["dev_labels"], sep="\t", encoding="utf-8", header=None
    )
    labels_dev = labels_dev.rename(columns={0: "id", 1: "line", 2: "labels"})
    labels_dev = labels_dev.set_index(["id", "line"])

    train = make_dataframe(paths["train_folder"], paths["train_labels"])
    dev = make_dataframe(paths["dev_folder"], paths["dev_labels"])
