import os
import sys
import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import pandas as pd

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers

# Load pre-trained model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
model = AutoModelForMaskedLM.from_pretrained("launch/POLITICS")

# Define the languages
languages = ["en", "fr", "ge", "it", "po", "ru"]

all_embeddings = []
all_labels = []


def get_embeddings(texts):
    embeddings = []
    for text in tqdm(texts):
        # Tokenize input and convert to PyTorch tensors
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Run the text through BERT
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Get the embeddings of the [CLS] token
        cls_embedding = outputs.last_hidden_state[0][0].numpy()

        embeddings.append(cls_embedding)

    return embeddings


def make_dataframe_template(path: str, labels_fn: str = None):
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
    df = df.set_index(["id", "line"])

    if labels_fn:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn, sep="\t", encoding="utf-8", header=None)
        labels = labels.rename(columns={0: "id", 1: "line", 2: "labels"})
        labels = labels.set_index(["id", "line"])
        labels = labels[labels.labels.notna()].copy()

        # JOIN
        df = labels.join(df)[["text", "labels"]]

    return df


for language in languages:
    # Define paths
    paths: dict = helpers.get_paths(language, "precomputed_POLITICS")

    translated_train = paths["train_template_translated"]
    labels_train_fn = paths["train_labels"]

    # Read Data
    print(f"Loading {language} train dataset...")
    train = bundle_baseline.make_dataframe(translated_train, labels_train_fn)

    X_train = train["text"].values
    Y_train = train["labels"].fillna("").str.split(",").values

    embeddings = get_embeddings(X_train)

    all_embeddings.extend(embeddings)
    all_labels.extend(Y_train)
    # Save the embeddings and labels to a file
    with open("all_train_cased_embeddings.pkl", "wb") as f:
        pickle.dump(all_embeddings, f)

    with open("all_train_cased_labels.pkl", "wb") as f:
        pickle.dump(all_labels, f)