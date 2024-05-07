from venv import logger
import numpy as np
import pandas as pd
from regex import F
import torch
import sys
import argparse
import os
import torch.nn as nn

from sklearn.metrics import classification_report as report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from torch.optim import Adam

sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file, evaluate
from bundle.baselines.st3 import *
from models.helpers import *  # myb switch to helper. use to make it clear the source of the function
from models.bert_baseline.bert_baseline import *


# def main():
#     parser = argparse.ArgumentParser(description="Subtask-3")
#     parser.add_argument(
#         "language",
#         type=str,
#         nargs=1,
#         help="Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)",
#     )
#     args = parser.parse_args()

#     language = args.language[0]
#     paths: dict = get_paths(language)

#     # TODO remove this later and replace with the use of the paths dictionary

#     out_fn = os.path.join(
#         BASE_PATH, f"outputs\\bert_baseline\\{language}_output.txt"
#     )
#     out_true = os.path.join(
#         BASE_PATH, f"outputs\\bert_baseline\\{language}_output_true.txt"
#     )

#     out_dev = os.path.join(
#         BASE_PATH, f"outputs\\bert_baseline\\{language}_dev_output.txt"
#     )
#     out_dev_true = os.path.join(
#         BASE_PATH, f"outputs\\bert_baseline\\{language}_dev_output_true.txt"
#     )

#     classes = (
#         CLASSES_SUBTASK_3_PATH  # this never changes so we can use the constant
#     )

#     labels = pd.read_csv(
#         paths["train_labels"], sep="\t", encoding="utf-8", header=None
#     )
#     labels = labels.rename(columns={0: "id", 1: "line", 2: "labels"})
#     labels = labels.set_index(["id", "line"])

#     labels_dev = pd.read_csv(
#         paths["dev_labels"], sep="\t", encoding="utf-8", header=None
#     )
#     labels_dev = labels_dev.rename(columns={0: "id", 1: "line", 2: "labels"})
#     labels_dev = labels_dev.set_index(["id", "line"])

#     train = make_dataframe(paths["train_folder"], paths["train_labels"])
#     dev = make_dataframe(paths["dev_folder"], paths["dev_labels"])

#     X_train = train["text"].values.tolist()
#     Y_train = (
#         train["labels"].fillna("").str.split(",").values
#     )  # possibly same as labels?

#     X_dev = dev["text"].values.tolist()
#     Y_dev = dev["labels"].fillna("").str.split(",").values

#     multibin = MultiLabelBinarizer()  # mayb needs classes as parameter?
#     Y_train = multibin.fit_transform(
#         Y_train
#     )  # or learns the classes from the data?
#     multibin_dev = MultiLabelBinarizer()  # obsolete?
#     Y_dev = multibin_dev.fit_transform(Y_dev)

#     tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#     model = BertModel.from_pretrained("bert-base-multilingual-cased")

#     loss_function = torch.nn.BCEWithLogitsLoss()
#     optimizer = Adam(model.parameters(), lr=1e-4)  # bert params??

#     batch_size = 16
#     total_loss = 0.0
#     num_epochs = 5
#     num_classes = 19
#     patience = 3
#     best_loss = float("inf")

#     for epoch in range(num_epochs):

#         with open(out_fn, "w") as file:
#             pass
#         with open(out_dev, "w") as file:
#             pass
#         with open(out_true, "w") as file:
#             pass
#         with open(out_dev_true, "w") as file:
#             pass

#         total_loss = 0.0
#         for i in range(0, 100, batch_size):
#             model.train()

#             batch_texts = X_train[i : i + batch_size]
#             tokenized_input = tokenizer(
#                 batch_texts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=300,
#             )

#             with torch.no_grad():
#                 outputs = model(**tokenized_input)

#             last_hidden_states = (
#                 outputs.last_hidden_state
#             )  # [batch_size, max. sequence_length, size of hidden layer]

#             Y_batch = Y_train[i : i + batch_size]
#             Y_labels = train[i : i + batch_size]

#             drop_layer = torch.nn.Dropout(0.3)
#             drop = drop_layer(last_hidden_states)
#             linear_layer = torch.nn.Linear(
#                 last_hidden_states.shape[-1], num_classes
#             )  # should exist before training!!
#             classification_output = linear_layer(drop)
#             print(
#                 classification_output.shape
#             )  # [batch_size, max. sequence_length, number of classes]

#             cl = classification_output[:, -1, :]
#             print(cl[0])
#             logits_S = torch.sigmoid(classification_output)
#             copy_sig = logits_S.round().int().squeeze(-1)

#             """
#             sigmoid_output = torch.sigmoid(classification_output)
#             copy_sig = sigmoid_output.clone()  # why clone?
#             copy_sig[copy_sig >= 0.5] = 1
#             copy_sig[copy_sig < 0.5] = 0
#             # print("Y_batch_one_hot_labels:", Y_batch)
#             # print(Y_batch.shape)
#             # print(sigmoid_output.shape)
#             """

#             sigmoid_o = copy_sig[
#                 :, 0, :
#             ]  # why is this 3rd order tensor? is it?
#             predicted_labels = sigmoid_o.int().tolist()

#             out = multibin.inverse_transform(torch.tensor(predicted_labels))
#             output = list(map(lambda x: ",".join(x), out))
#             ou = pd.DataFrame(output, Y_labels.index)

#             ou.to_csv(out_fn, sep="\t", header=None, mode="a")
#             # ou.loc[first_indx:last_processed_index+1].to_csv(out_fn, sep='\t', header=None, mode='a')

#             f1_macro = f1_score(
#                 Y_batch, predicted_labels, average="macro", zero_division=0
#             )
#             f1_micro = f1_score(
#                 Y_batch, predicted_labels, average="micro", zero_division=0
#             )
#             print("F1 Macro:", f1_macro)
#             print("F1 Micro:", f1_micro)

#             loss = loss_function(
#                 classification_output[:, -1, :],
#                 torch.tensor(Y_batch, dtype=torch.float32),
#             )  # micanje srednje dim. da dobijemo velicinu najdulje recenice

#             print("Batch loss:", loss, " for ", i)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         ou = pd.read_csv(out_fn, sep="\t", encoding="utf-8", header=None)
#         ou = ou.rename(columns={0: "id", 1: "line", 2: "labels"})
#         ou = ou.set_index(["id", "line"])

#         copied_labels = labels.copy()
#         for index in labels.index:
#             mask = ou.index == index
#             if mask.any():
#                 labels.at[index, "labels"] = ou[mask].values[0]
#             else:
#                 labels.at[index, "labels"] = ""
#             if index == ou.index[-1]:
#                 copied_labels = copied_labels.loc[:index].copy()
#                 copied_labels.to_csv(out_true, sep="\t", header=None)
#                 break

#         labels.loc[:index].copy().to_csv(out_fn, sep="\t", header=None)

#         average_loss = total_loss / (100 // batch_size)
#         print("Epoch:", epoch + 1, "Average Batch Loss:", average_loss)

#         """
#         pred_labels = _read_csv_input_file(out_fn)
#         true_labels = _read_csv_input_file(out_true)
#         macro_f1, micro_f1 = evaluate(pred_labels, true_labels, classes)
#         print("Epoch:", epoch + 1, "F1 macro:", macro_f1, "F1 micro:", micro_f1)
#         """

#         dev_loss = 0.0
#         model.eval()
#         for i in range(0, 100, batch_size):
#             Y_labels = dev[i : i + batch_size]

#             batch_texts = X_dev[i : i + batch_size]
#             tokenized_input = tokenizer(
#                 batch_texts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=300,
#             )

#             outputs = model(**tokenized_input)
#             last_hidden_states = outputs.last_hidden_state

#             linear_layer = torch.nn.Linear(last_hidden_states.shape[-1], 19)
#             classification_output = linear_layer(last_hidden_states)

#             sigmoid_output2 = torch.sigmoid(classification_output)
#             copy_sig2 = sigmoid_output2.clone()
#             copy_sig2[copy_sig2 >= 0.5] = 1
#             copy_sig2[copy_sig2 < 0.5] = 0

#             sigmoid_o2 = copy_sig2[:, 0, :]

#             predicted_labels_dev = sigmoid_o2.int().tolist()
#             print("predicted labels: ", predicted_labels_dev)
#             out_de = multibin_dev.inverse_transform(
#                 torch.tensor(predicted_labels_dev)
#             )
#             # out_de = multibin_dev.inverse_transform(sigmoid_o2.detach().numpy())
#             output_dev = list(map(lambda x: ",".join(x), out_de))
#             ou_dev = pd.DataFrame(output_dev, Y_labels.index)
#             ou_dev.to_csv(out_dev, sep="\t", header=None, mode="a")

#             Y_devv = Y_dev[i : i + batch_size]
#             Y_devv_float64 = np.array(
#                 [np.array(y, dtype=np.float64) for y in Y_devv]
#             )
#             loss = loss_function(
#                 sigmoid_o2, torch.tensor(Y_devv_float64, dtype=torch.float64)
#             )

#             dev_loss += loss.item()

#         ou_dev = pd.read_csv(out_dev, sep="\t", encoding="utf-8", header=None)
#         ou_dev = ou_dev.rename(columns={0: "id", 1: "line", 2: "labels"})
#         ou_dev = ou_dev.set_index(["id", "line"])

#         # print(ou_dev) # is this reconstruction of the template?
#         # a helper function may be useful for this
#         copied_labels_dev = labels_dev.copy()
#         for index_dev in labels_dev.index:
#             mask_dev = ou_dev.index == index_dev
#             if mask_dev.any():
#                 labels_dev.at[index_dev, "labels"] = ou_dev[mask_dev].values[0]
#             else:
#                 labels_dev.at[index_dev, "labels"] = ""
#             if index_dev == ou_dev.index[-1]:
#                 copied_labels_dev = copied_labels_dev.loc[:index_dev].copy()
#                 copied_labels_dev.to_csv(out_dev_true, sep="\t", header=None)
#                 break

#         # labels.to_csv(out_fn, sep='\t', header=None, mode='a') # mode = 'a' ako zelimo nastaviti pisati odakle smo stali u .txt
#         labels_dev.loc[:index_dev].copy().to_csv(out_dev, sep="\t", header=None)

#         pred_labels = _read_csv_input_file(out_dev)
#         true_labels = _read_csv_input_file(out_dev_true)
#         macro_f1, micro_f1 = evaluate(pred_labels, true_labels, classes)
#         print("Epoch:", epoch + 1, "F1 macro:", macro_f1, "F1 micro:", micro_f1)

#         average_dev_loss = dev_loss / (len(X_dev) // batch_size)

#         print(
#             f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss}, Dev Loss: {average_dev_loss}"
#         )

#         if average_dev_loss < best_loss:
#             best_loss = average_dev_loss
#             early_stopping_counter = 0
#         else:
#             early_stopping_counter += 1

#         if early_stopping_counter >= patience and epoch < num_epochs - 1:
#             print(
#                 "Validation loss did not improve for",
#                 patience,
#                 "epochs. Early stopping...",
#             )
#             break

#     average_loss = total_loss / (len(X_train) // batch_size)
#     print("Average Batch Loss:", average_loss)

#     return average_loss


if __name__ == "__main__":
    second_main()
