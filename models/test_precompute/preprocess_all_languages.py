import os
import sys
import pickle
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory

import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers

# Load pre-trained model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

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


for language in languages:
    # Define paths
    paths: dict = helpers.get_paths(language, "precompute_bert")

    folder_train = paths["train_folder"]
    labels_train_fn = paths["train_labels"]

    # Read Data
    print(f"Loading {language} train dataset...")
    train = bundle_baseline.make_dataframe(folder_train, labels_train_fn)

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
