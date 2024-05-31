import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)
sys.path.append(BASE_PATH)  # enable importing from the root directory

import models.helpers as helpers


class PrecomputedEmbeddings(Dataset):
    def __init__(
        self,
        embeddings_model_name: str,
        languages: list,
        split_type: str,
    ):
        embeddings = pd.DataFrame()
        labels = pd.DataFrame()

        with open(helpers.CLASSES_SUBTASK_3_PATH, "r") as f:
            classes = f.read().split("\n")
            classes = [c for c in classes if c != ""]
        self.labels_transformer = MultiLabelBinarizer(classes=classes)

        for lang in languages:
            paths = helpers.get_paths(lang, embeddings_model_name)
            with open(paths[f"{split_type}_embeddings"], "rb") as f:
                embeddings = pd.concat([embeddings, pd.read_pickle(f)])

            lang_labels = pd.read_csv(
                paths[f"{split_type}_labels"],
                sep="\t",
                encoding="utf-8",
                header=None,
            ).rename(columns={0: "id", 1: "line", 2: "labels"})

            labels = pd.concat([labels, lang_labels])

        labels["labels"] = labels["labels"].fillna("")
        labels["labels"] = labels["labels"].str.split(",")

        self.embeddings = embeddings
        self.labels = labels
        data = pd.merge(embeddings, labels, on=["id", "line"])
        self.data = data
        # print("Data rows:", len(data)) # TODO fix this in dataset
        # print("embeddings rows:", len(embeddings))
        # print("Labels rows:", len(labels))

        # # labels rows not in embeddings
        # missing_ids = []
        # for idx, row in labels.iterrows():
        #     if len(
        #         embeddings[
        #             (embeddings["id"] == row["id"])
        #             & (embeddings["line"] == row["line"])
        #         ]
        #     ):
        #         continue

        #     missing_ids.append(row["id"])
        # print("Missing embeddings rows:")
        # print(missing_ids)
        # print("Missing embeddings rows count:", len(missing_ids))

        data["multi_hot_labels"] = self.labels_transformer.fit_transform(
            data["labels"]
        ).tolist()  # docs say If the classes parameter is set, y will not be iterated.

        # print("Data rows after transformation:", len(data))
        # print("Data columns:", data.columns)
        # print("Data head:", data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Function to retrieve the data sample at the specified index.

        Args:
        ------
            idx (int): The index of the data sample.

        Returns:
        --------
            embeddings (torch.Tensor): Tensor containing the precomputed embeddings of the text.
            labels (torch.Tensor): Tensor containing the true multiple-hot encoded labels.
        """
        id = self.data.iloc[idx]["id"]
        line = self.data.iloc[idx]["line"]
        embeddings = torch.tensor(self.data.iloc[idx]["embeddings"])
        hot_labels = torch.tensor(
            self.data.iloc[idx]["multi_hot_labels"], dtype=torch.float32
        )

        return id, line, embeddings, hot_labels


class PrecomputedEmbeddingsAndST2Labels(Dataset):
    def __init__(
        self,
        embeddings_model_name: str,
        languages: list,
        split_type: str,
    ):
        embeddings = pd.DataFrame()
        labels = pd.DataFrame()
        st2_labels = pd.DataFrame()

        with open(helpers.FRAMES_SUBTASK_2_PATH, "r") as f:
            frames = f.read().split("\n")
            frames = [c for c in frames if c != ""]
        self.frames_transformer = MultiLabelBinarizer(classes=frames)

        with open(helpers.CLASSES_SUBTASK_3_PATH, "r") as f:
            classes = f.read().split("\n")
            classes = [c for c in classes if c != ""]
        self.labels_transformer = MultiLabelBinarizer(classes=classes)

        for lang in languages:
            paths = helpers.get_paths(lang, embeddings_model_name)
            with open(paths[f"{split_type}_embeddings"], "rb") as f:
                embeddings = pd.concat([embeddings, pd.read_pickle(f)])

            lang_labels = pd.read_csv(
                paths[f"{split_type}_labels"],
                sep="\t",
                encoding="utf-8",
                header=None,
            ).rename(columns={0: "id", 1: "line", 2: "labels"})

            labels = pd.concat([labels, lang_labels])

            paths_st2 = helpers.get_paths(lang, embeddings_model_name, 2)
            lang_st2_labels = pd.read_csv(
                paths_st2[f"{split_type}_labels"],
                sep="\t",
                encoding="utf-8",
                header=None,
            ).rename(columns={0: "id", 1: "frames"})
            st2_labels = pd.concat([st2_labels, lang_st2_labels])

        st2_labels["frames"] = st2_labels["frames"].fillna("")
        st2_labels["frames"] = st2_labels["frames"].str.split(",")

        labels["labels"] = labels["labels"].fillna("")
        labels["labels"] = labels["labels"].str.split(",")

        self.frames = st2_labels
        self.embeddings = embeddings
        self.labels = labels
        data = pd.merge(embeddings, labels, on=["id", "line"])
        data = pd.merge(data, st2_labels, on=["id"])
        self.data = data
        self.embeddings_dimension = len(data.iloc[0]["embeddings"])

        data["multi_hot_labels"] = self.labels_transformer.fit_transform(
            data["labels"]
        ).tolist()  # docs say If the classes parameter is set, y will not be iterated.

        data["multi_hot_frames"] = self.frames_transformer.fit_transform(
            data["frames"]
        ).tolist()

        # print("Data rows after transformation:", len(data))
        # print("Data columns:", data.columns)
        # print("Data head:", data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Function to retrieve the data sample at the specified index.

        Args:
        ------
            idx (int): The index of the data sample.

        Returns:
        --------
            embeddings (torch.Tensor): Tensor containing the precomputed embeddings of the text.
            labels (torch.Tensor): Tensor containing the true multiple-hot encoded labels.
        """
        id = self.data.iloc[idx]["id"]
        line = self.data.iloc[idx]["line"]
        embeddings = torch.tensor(self.data.iloc[idx]["embeddings"])
        hot_labels = torch.tensor(
            self.data.iloc[idx]["multi_hot_labels"], dtype=torch.float32
        )
        hot_frames = torch.tensor(
            self.data.iloc[idx]["multi_hot_frames"], dtype=torch.float32
        )

        return id, line, embeddings, hot_labels, hot_frames


if __name__ == "__main__":
    # Test the PrecomputedEmbeddings class
    model_name = "bert-base-multilingual-cased"
    languages = helpers.languages_train
    split_type = "dev"
    dataset = PrecomputedEmbeddings(model_name, languages, split_type)
    print(dataset[1])

    # Test the PrecomputedEmbeddingsAndST2Labels class
    print("----------------------------------------------")
    print("Testing PrecomputedEmbeddingsAndST2Labels")
    dataset = PrecomputedEmbeddingsAndST2Labels(
        model_name, languages, split_type
    )
