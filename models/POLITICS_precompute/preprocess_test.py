from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pickle
import sys
from tqdm import tqdm
import pandas as pd

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
print("Tokenizer loaded")
model = AutoModelForMaskedLM.from_pretrained("launch/POLITICS")
print("Model loaded")


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
    df = df.set_index(["id", "line"])
    return df


# Define paths
paths: dict = helpers.get_paths("en", "precomputed_POLITICS")

translated_dev = paths["dev_template_translated"]

# Read Data
print("Loading dataset...")
test = make_dataframe_template(translated_dev)

X_test = test["text"].values


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


print(X_test)
embeddings_test = get_embeddings(X_test)
# Save the embeddings to a file
with open("embeddings_en_test.pkl", "wb") as f:
    pickle.dump(embeddings_test, f)
