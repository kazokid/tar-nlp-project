from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset
import sys

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)
sys.path.append(BASE_PATH)  # enable importing from the root directory

import models.helpers as helpers


class PrecomputedEmbeddingsAndST2Labels(Dataset):
    def __init__(
        self,
        embeddings_model_name: str,
        languages: list,
        split_type: str,
    ):
        with open(helpers.FRAMES_SUBTASK_2_PATH, "r") as f:
            frames = f.read().split("\n")
            frames = [c for c in frames if c != ""]
        self.frames_transformer = MultiLabelBinarizer(classes=frames)

        with open(helpers.CLASSES_SUBTASK_3_PATH, "r") as f:
            classes = f.read().split("\n")
            classes = [c for c in classes if c != ""]
        self.labels_transformer = MultiLabelBinarizer(classes=classes)

        if split_type in ["validation", "test"]:
            path_split_type = "dev"
        else:
            path_split_type = split_type

        self.data = pd.DataFrame()

        for lang in languages:
            paths = helpers.get_paths(lang, embeddings_model_name)
            with open(paths[f"{path_split_type}_embeddings"], "rb") as f:
                lang_embeddings = pd.read_pickle(f)

            lang_labels = pd.read_csv(
                paths[f"{path_split_type}_labels"],
                sep="\t",
                encoding="utf-8",
                header=None,
            ).rename(columns={0: "id", 1: "line", 2: "labels"})

            lang_labels["labels"] = lang_labels["labels"].fillna("")
            lang_labels["labels"] = lang_labels["labels"].str.split(",")

            paths_st2 = helpers.get_paths(lang, embeddings_model_name, 2)
            lang_frames = pd.read_csv(
                paths_st2[f"{path_split_type}_labels"],
                sep="\t",
                encoding="utf-8",
                header=None,
            ).rename(columns={0: "id", 1: "frames"})
            lang_frames["frames"] = lang_frames["frames"].fillna("")
            lang_frames["frames"] = lang_frames["frames"].str.split(",")

            # merge and split at lanuage level ensure that there is no leakage between validation
            # and test when loading different language lists
            lang_data = pd.merge(
                lang_embeddings, lang_labels, on=["id", "line"]
            )
            lang_data = pd.merge(lang_data, lang_frames, on=["id"])

            if split_type in ["validation", "test"]:
                valid_data, test_data = train_test_split(
                    lang_data, test_size=0.5, random_state=42
                )

                if split_type == "validation":
                    lang_data = valid_data
                elif split_type == "test":
                    lang_data = test_data

            self.data = pd.concat([self.data, lang_data])

        self.embeddings_dimension = len(self.data.iloc[0]["embeddings"])

        self.data["multi_hot_labels"] = self.labels_transformer.fit_transform(
            self.data["labels"]
        ).tolist()  # docs say If the classes parameter is set, y will not be iterated.

        self.data["multi_hot_frames"] = self.frames_transformer.fit_transform(
            self.data["frames"]
        ).tolist()

        # print("Data rows after transformation:", len(data))
        # print("Data columns:", data.columns)
        # print("Data head:", data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
    # Test the PrecomputedEmbeddingsAndST2Labels class
    print("----------------------------------------------")
    print("Testing PrecomputedEmbeddingsAndST2Labels")
    dataset_valid = PrecomputedEmbeddingsAndST2Labels(
        model_name, languages, "validation"
    )
    dataset_test = PrecomputedEmbeddingsAndST2Labels(
        model_name, languages, "test"
    )
    dataset_train = PrecomputedEmbeddingsAndST2Labels(
        model_name, ["en"], "train"
    )
