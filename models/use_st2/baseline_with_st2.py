import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn

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


class BaselineWST2(nn.Module):
    def __init__(
        self,
        embeddings_dimension: int,
        num_classes: int,
        num_frames: int,
        drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(
            embeddings_dimension + num_frames, num_classes
        )

    def forward(self, x):
        x = self.linear1(x)

        return x


class Baseline2LayerWST2(nn.Module):
    def __init__(
        self,
        embeddings_dimension: int,
        num_classes: int,
        num_frames: int,
        drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(embeddings_dimension + num_frames, 300)
        self.drop = torch.nn.Dropout(drop)
        self.linear2 = torch.nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


class Baseline3LayerWST2(nn.Module):
    def __init__(
        self,
        embeddings_dimension: int,
        num_classes: int,
        num_frames: int,
        drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(embeddings_dimension + num_frames, 300)
        self.drop1 = torch.nn.Dropout(drop)
        self.linear2 = torch.nn.Linear(300, 100)
        self.drop2 = torch.nn.Dropout(drop)
        self.linear3 = torch.nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = torch.relu(x)
        x = self.linear3(x)

        return x


class Baseline4LayerWST2(nn.Module):
    def __init__(
        self,
        embeddings_dimension: int,
        num_classes: int,
        num_frames: int,
        drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(embeddings_dimension + num_frames, 300)
        self.drop1 = torch.nn.Dropout(drop)
        self.linear2 = torch.nn.Linear(300, 100)
        self.drop2 = torch.nn.Dropout(drop)
        self.linear3 = torch.nn.Linear(100, 50)
        self.drop3 = torch.nn.Dropout(drop)
        self.linear4 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = self.drop3(x)
        x = torch.relu(x)
        x = self.linear4(x)

        return x


class Baseline5LayerWST2(nn.Module):
    def __init__(
        self,
        embeddings_dimension: int,
        num_classes: int,
        num_frames: int,
        drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(embeddings_dimension + num_frames, 300)
        self.drop1 = torch.nn.Dropout(drop)
        self.linear2 = torch.nn.Linear(300, 100)
        self.drop2 = torch.nn.Dropout(drop)
        self.linear3 = torch.nn.Linear(100, 50)
        self.drop3 = torch.nn.Dropout(drop)
        self.linear4 = torch.nn.Linear(50, 30)
        self.drop4 = torch.nn.Dropout(drop)
        self.linear5 = torch.nn.Linear(30, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = self.drop3(x)
        x = torch.relu(x)
        x = self.linear4(x)
        x = self.drop4(x)
        x = torch.relu(x)
        x = self.linear5(x)

        return x


def train(model, optimizer, criterion, train_loader):
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()

        ids, lines, embeddings, hot_labels, hot_frames = batch

        combined = torch.cat((embeddings, hot_frames), dim=1)

        logits = model(combined)
        loss = criterion(logits, hot_labels)

        loss.backward()
        optimizer.step()


def evaluate(
    model,
    criterion,
    val_loader: DataLoader[
        dataset_preparation.PrecomputedEmbeddingsAndST2Labels
    ],
    threshold=0.3,
):
    model.eval()
    predicted_labels = {}
    true_labels = {}
    labels_tramsformer: MultiLabelBinarizer = (
        val_loader.dataset.labels_transformer
    )

    with torch.no_grad():
        for batch in val_loader:
            ids, lines, embeddings, labels, frames = batch

            combined = torch.cat((embeddings, frames), dim=1)

            logits = model(combined)
            prediction = (torch.sigmoid(logits) > threshold).int()
            predicted_text = labels_tramsformer.inverse_transform(
                prediction.cpu().numpy()
            )
            true_text = labels_tramsformer.inverse_transform(
                labels.cpu().numpy()
            )

            for id, line, pred, true in zip(
                ids, lines, predicted_text, true_text
            ):

                predicted_labels[f"{id}_{line}"] = pred
                true_labels[f"{id}_{line}"] = true

    macro_f1, micro_f1 = bundle_scorer.evaluate(
        predicted_labels, true_labels, labels_tramsformer.classes
    )

    return macro_f1, micro_f1


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_datataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "train"
    )
    val_dataset = dataset_preparation.PrecomputedEmbeddingsAndST2Labels(
        "bert-base-multilingual-cased", helpers.languages_train, "dev"
    )

    train_loader = DataLoader(train_datataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = Baseline2LayerWST2(
        768,
        len(train_datataset.labels_transformer.classes),
        len(train_datataset.frames_transformer.classes),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    patience = 5
    best_micro = 0
    for epoch in range(int(1e3)):
        train(model, optimizer, criterion, train_loader)
        macro_f1, micro_f1 = evaluate(model, criterion, val_loader)

        print(f"Epoch {epoch + 1}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        if micro_f1 > best_micro:
            best_micro = micro_f1
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            break
    return


if __name__ == "__main__":
    main()
