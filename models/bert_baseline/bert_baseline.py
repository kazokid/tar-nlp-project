import numpy as np
import pandas as pd
import torch
import sys
import argparse
import os
import torch.nn as nn

from sklearn.metrics import classification_report as accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from bundle.scorers.scorer_subtask_3 import _read_csv_input_file

sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import *
from bundle.baselines.st3 import *
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function
from models.model_preparation import *


class BertBaseline(nn.Module):
    def __init__(self, language: str, num_classes: int) -> None:
        super().__init__()
        self.language = language
        self.num_classes = num_classes
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.bert_model = BertModel.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.loss_function = (
            torch.nn.BCEWithLogitsLoss()
        )  # check if this is the correct loss function

        self.drop = torch.nn.Dropout(
            0.3
        )  # 0.3 - dont know what actually 0.3 means, one team has set the value to that
        self.linear1 = torch.nn.Linear(
            self.bert_model.config.hidden_size, self.num_classes
        )

    def added_parameters(self):
        """Returns the parameters of the model that need to be optimized - excluding the BERT model parameters"""
        return list(self.linear1.parameters())

    def forward(self, x):

        input_ids, attention_mask, labels = x

        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        last_hidden_states = (
            outputs.last_hidden_state
        )  # [batch_size, max. sequence_length, size of hidden layer]
        print("Last hidden state shape: ", last_hidden_states.shape)

        drop = self.drop(last_hidden_states)
        linear_output = self.linear1(drop)
        print("Linear output shape: ", linear_output.shape)

        return linear_output


def model_pass(
    model,
    data_loader,
    indexes,
    multibin_decode,
    threshold,
    out_file_predictions,
    out_f_metrics,
    bool_train,
    params=list(),
    n_epochs=5,
    lr=1e-05,
):
    """
    Function to train or evaluate a model on a dataset and make predictions.

    Args:
    ------
    model (nn.Module):                      The model to train or evaluate.
    data_loader (DataLoader):               DataLoader object containing the dataset.
    indexes (Multilabel):                   Indexes of sentences for which predictions are made (article_id, row_id).
    multibin_decode (MultiLabelBinarizer):  MultiLabelBinarizer object for decoding one-hot vectors to label predictions.
    threshold (float):                      Threshold for sigmoid function.
    out_file_predictions (file):            File to write the predictions.
    out_f_metrics (file):                   File to write the evaluation metrics.
    bool_train (bool):                      Flag indicating whether to train the model (True) or evaluate (False).
    params (list):                          Model parameters to optimize (default=[]).
    n_epochs (int):                         Number of epochs for training (default=5).
    lr (float):                             Learning rate for optimizer (default=1e-05).

    Returns:
    --------
    None
    """

    with open(out_file_predictions, "w") as f_predictions:
        pass
    with open(out_f_metrics, "w") as f_metrics:
        f_metrics.write("Epoch\tAccuracy\tF1 Macro\tF1 Micro\n")

    if bool_train == True:
        model.train()  # set model to training mode, possibly useless (no dropout or batchnorm layers in this model)
    else:
        model.eval()

    optimizer = Adam(params, lr=lr)
    for epoch in range(n_epochs):
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        total_examples_processed = 0

        for batch_num, batch in enumerate(data_loader):

            if bool_train == True:
                optimizer.zero_grad()

            input_ids, attention_mask, true_labels = batch

            Y_batch = true_labels.clone().detach().to(torch.float32)
            batch_indexes = indexes[
                batch_num : batch_num + data_loader.batch_size
            ]

            logits = model.forward(
                (input_ids, attention_mask, None)
            )  # output dim = [batch_size, max. sequence_length, number of classes]
            print("Logits shape: ", logits.shape)
            print("Labels shape: ", Y_batch.shape)
            loss = model.loss_function(logits[:, -1, :], Y_batch)

            if bool_train == True:
                loss.backward()
                optimizer.step()

            sigmoid_output = torch.sigmoid(logits[:, -1, :])
            sigmoid_output[sigmoid_output >= threshold] = 1
            sigmoid_output[sigmoid_output < threshold] = 0

            predictions = multibin_decode.inverse_transform(
                sigmoid_output.detach().numpy()
            )
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)

            pred_list = list(map(lambda x: ",".join(x), predictions))
            df_pred = pd.DataFrame(pred_list, batch_indexes)
            df_pred.to_csv(
                out_file_predictions, sep="\t", header=None, mode="a"
            )

            total_loss += loss.item()
            total_examples_processed += len(input_ids)

            print(
                f"Epoch {epoch + 1}, Batch {batch_num + 1}, Examples Processed: {total_examples_processed}"
            )

            print()

        average_loss = total_loss / (
            len(data_loader.dataset) // data_loader.batch_size
        )
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average="macro")
        f1_micro = f1_score(all_labels, all_predictions, average="micro")

        out_f_metrics.write(
            f"{epoch + 1}\t{accuracy}\t{f1_macro}\t{f1_micro}\n"
        )
        # Is there a need to calculate the loss? -> I didn't write it down in .txt
        print(
            "Epoch:",
            epoch + 1,
            "Average Batch Loss:",
            average_loss,
            "Accuracy:",
            accuracy,
            "F1 Macro:",
            f1_macro,
            "F1 Micro:",
            f1_micro,
        )


# ne triba nam ovo?
def predict(model, X, threshold=0.5):
    model.eval()  # set model to evaluation mode, possibly useless (no dropout or batchnorm layers in this model)

    with torch.no_grad():
        clasification_measures = model.forward(X)
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

    # todo move this to the helper method get_paths since it's a constant
    parser.add_argument(
        "techniques_file_path",
        type=str,
        help="Path to the file with the names of the techniques",
    )

    args = parser.parse_args()
    language = args.language[0]

    paths: dict = get_paths(language)

    CLASSES = read_techniques_list_from_file(CLASSES_SUBTASK_3_PATH)

    # label loading
    train_labels = pd.read_csv(
        paths["train_labels"], sep="\t", encoding="utf-8", header=None
    )
    train_labels = train_labels.rename(
        columns={0: "id", 1: "line", 2: "labels"}
    )
    train_labels = train_labels.set_index(["id", "line"])

    dev_labels = pd.read_csv(
        paths["dev_labels"], sep="\t", encoding="utf-8", header=None
    )
    dev_labels = dev_labels.rename(columns={0: "id", 1: "line", 2: "labels"})
    dev_labels = dev_labels.set_index(["id", "line"])

    train = make_dataframe(paths["train_folder"], paths["train_labels"])
    dev = make_dataframe(paths["dev_folder"], paths["dev_labels"])

    print("train['text']:\n", train["text"])
    print()
    print("train['labels']\n", train["labels"])
    print()
    # make_dataframe - vraca df gdje imamo 'text' i 'labels' columns
    # train['text'] -> sastoji se od multilabel index-a (article_id, row_id), a train['text'].values - nas tekst
    # ista stvar vrijedi za train['labels']

    # TODO
    # dataloader loading
    # vidjeti je li bolje raditi sa funkcijom create_dataloader ili ubaciti te 3 linije koda tu
    # prilikom kreiranja CustomDataset i koristenja tokenizer-a - vidjeti najbolji max length koji bi bio dobar za uzeti!!!

    train_dataloader, train_hot_encoder = create_dataloader(
        train, "bert-base-multilingual-cased"
    )
    train_indx = (
        train_dataloader.dataset.data.index
    )  # dohvacanje indexa (article id, santence id) kako bi mogli spremiti predikcije za odredeni indeks

    dev_dataloader, dev_hot_encoder = create_dataloader(
        dev, "bert-base-multilingual-cased"
    )
    dev_indx = dev_dataloader.dataset.data.index

    num_classes = len(train_hot_encoder.classes_)
    model = BertBaseline("en", num_classes)
    params = model.added_parameters()

    # training
    model_pass(
        model,
        train_dataloader,
        train_indx,
        train_hot_encoder,
        0.5,
        paths["train_predictions"],
        paths["train_metrics"],
        True,
        params,
    )

    # evaluating
    pred_labels_train = _read_csv_input_file(paths["train_predictions"])
    gold_labels_train = _read_csv_input_file(paths["train_labels"])

    # TODO
    # nisam pokrenila model za cijeli dataset 'en'
    # vidjeti je li donje radi
    # pred_labels_train - u ovom slucaju ce sadrzavati samo red za redom
    #                       tj samo one redove za koje je model predvidio labelu
    #                       nece sadrzavati prazne redove tj redove sa indeksima za koje predikcija ne postoji
    #                       ? Mozda ce za evaluate trebati narpaviti file i sa redovima za koje predikcija labele ne postoji?
    evaluate(pred_labels_train, gold_labels_train, CLASSES)

    # testing
    model_pass(
        model,
        dev_labels,
        dev_dataloader,
        dev_indx,
        dev_hot_encoder,
        0.5,
        paths["dev_predictions"],
        paths["dev_metrics"],
        False,
    )

    # evaluating
    pred_labels_dev = _read_csv_input_file(paths["dev_predictions"])
    gold_labels_dev = _read_csv_input_file(paths["dev_labels"])
    evaluate(pred_labels_dev, gold_labels_dev, CLASSES)
