import torch
import pickle
import sys
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers
from transformers import BertTokenizer, BertModel
from models.baseline.baseline import Baseline2Layer
from models.use_st2_late_fuse.baseline_with_st2_late_fuse import (
    Baseline2LayerWST2LateFuse,
)

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


PARAMS_PATH = "hyperparam_test/Run_1/{model_name}.pth"

baseline_model = Baseline2Layer(768, 23)
baseline_model.load_state_dict(
    torch.load(PARAMS_PATH.format(model_name="baseline2layer"))
)

late_fuse_model = Baseline2LayerWST2LateFuse(768, 23, 14)
late_fuse_model.load_state_dict(
    torch.load(PARAMS_PATH.format(model_name="late_fuse2layer"))
)

with open(helpers.FRAMES_SUBTASK_2_PATH, "r") as f:
    frames = f.read().split("\n")
    frames = [c for c in frames if c != ""]
frames_transformer = MultiLabelBinarizer(classes=frames)
frames_transformer.fit([frames])

with open(helpers.CLASSES_SUBTASK_3_PATH, "r") as f:
    classes = f.read().split("\n")
    classes = [c for c in classes if c != ""]
labels_transformer = MultiLabelBinarizer(classes=classes)
labels_transformer.fit([classes])


def get_prediction_baseline(text):
    global labels_transformer, baseline_model
    embedding = get_embedding(text)
    with torch.no_grad():
        logits = baseline_model(torch.tensor(embedding).unsqueeze(0))
        probs = torch.sigmoid(logits)
        prediction = (probs > 0.2).int()
        predicted_texts = labels_transformer.inverse_transform(
            prediction.cpu().numpy()
        )

    return predicted_texts[0], probs[prediction == 1].numpy()


user_input = input("Enter a sentence: ")

while user_input != "exit":
    predictions, probs = get_prediction_baseline(user_input)
    if len(predictions) > 0:
        for label, prob in zip(predictions, probs):
            print(f"{label} ({prob:.2f})")
    else:
        print("No labels predicted")

    print("-" * 50)
    user_input = input("Enter a sentence: ")
