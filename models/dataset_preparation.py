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

        print("Data rows after transformation:", len(data))
        print("Data columns:", data.columns)
        print("Data head:", data.head())

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


# This should be done when training and evaluating the model
# def create_dataloader(data, model_name, batch_size=16, max_length=200):

#     dataset = CustomDataset(data, model_name, max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return dataloader, dataset.multi_label_binarizer()


if __name__ == "__main__":
    # Test the PrecomputedEmbeddings class
    model_name = "bert-base-multilingual-cased"
    languages = helpers.languages_train
    split_type = "dev"
    dataset = PrecomputedEmbeddings(model_name, languages, split_type)
    print(dataset[1])
