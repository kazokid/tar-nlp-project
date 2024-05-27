from transformers import RobertaTokenizer, RobertaModel
import torch
import pickle
import os
import sys
from tqdm import tqdm


sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory


import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers

# Load pre-trained model and tokenizer
model_name = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Define paths
paths: dict = helpers.get_paths("en", "RoBERTA_large")

folder_train = paths["train_folder"]
folder_dev = paths["dev_folder"]
labels_train_fn = paths["train_labels"]
out_fn = paths["dev_predictions"]

# Read Data
print("Loading dataset...")
train = bundle_baseline.make_dataframe(folder_train, labels_train_fn)
test = bundle_baseline.make_dataframe(folder_dev)

X_train = train["text"].values

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


embeddings_train = get_embeddings(X_train)
with open("embeddings_en_train.pkl", "wb") as f:
    pickle.dump(embeddings_train, f)

#embeddings_test = get_embeddings(X_test)
# Save the embeddings to a file
#with open("embeddings_en_RoBERTA_dev.pkl", "wb") as f:
#    pickle.dump(embeddings_test, f)