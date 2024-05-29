import torch
import pickle
import sys
import os
from tqdm import tqdm
import pandas as pd

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers
from transformers import BertTokenizer, BertModel

# CHECK THIS:
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

OUR_MODEL_NAME = "bert-base-multilingual-cased"


def get_embedding(text):
    # modify to handle longer texts if needed
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # Get the embeddings of the [CLS] token
    cls_embedding = outputs.last_hidden_state[0][0].numpy()

    return cls_embedding


def make_dataframe_template(path: str):
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
    return df


languages_train = ["en", "fr", "ge", "it", "po", "ru"]
languages_only_test = ["es", "gr", "ka"]
all_languages = languages_train + languages_only_test

data_type = ["train", "dev", "test"]

for lang in ["en"]:
    for type in data_type:
        if lang in languages_only_test and type != "test":
            continue
        paths = helpers.get_paths(lang, OUR_MODEL_NAME)

        # adjust this to use translated
        df = make_dataframe_template(paths[f"{type}_template"])

        os.makedirs(
            f"{BASE_PATH}/precomputed_embeddings/{OUR_MODEL_NAME}/{lang}/",
            exist_ok=True,
        )
        embeddings = pd.DataFrame()

        for i in tqdm(range(0, len(df), 100)):
            batch = df.iloc[i : i + 100].copy()

            batch["embeddings"] = batch["text"].apply(get_embedding)

            embeddings = pd.concat([embeddings, batch], ignore_index=True)

        # Save the embeddings to a file
        with open(
            f"{BASE_PATH}/precomputed_embeddings/{OUR_MODEL_NAME}/{lang}/{type}_embeddings.pkl",
            "wb",
        ) as f:
            pickle.dump(embeddings, f)
