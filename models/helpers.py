import os
import logging
import pandas as pd
from tqdm import tqdm
import os
import torch

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

# BASE_PATH = os.getenv("MY_REPO_LOCATION") #this should be removed if the above works
# if BASE_PATH is None:
#     raise ValueError("Please set the environment variable MY_REPO_LOCATION")

languages_train = ["fr", "ge", "it", "po", "ru"]
languages_only_test = ["es", "gr", "ka"]
all_languages = languages_train + languages_only_test

CLASSES_SUBTASK_3_PATH = os.path.join(
    BASE_PATH, "bundle/scorers/techniques_subtask3.txt"
)


def get_labels_folder_path(language, type, subtask):
    """Function that constructs the path to the folder given language and type (train, dev, test)

    Args:
    ------
        language (str): Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)
        type (str): Type of the dataset (train, dev, test)
        subtask (int): Subtask of the dataset (default is 3)

    Returns:
    --------
        path (str): Path to the folder with the labels of the given language and type
    """

    return os.path.join(
        BASE_PATH, f"bundle/data/{language}/{type}-labels-subtask-{subtask}/"
    )


def get_articles_folder_path(language, type, subtask):
    """Function that constructs the path to the folder given language and type (train, dev, test)

    Args:
    ------
        language (str): Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)
        type (str): Type of the dataset (train, dev, test)
        subtask (int): Subtask of the dataset (default is 3)

    Returns:
    --------
        path (str): Path to the folder with the articles of the given language and type
    """

    return os.path.join(
        BASE_PATH, f"bundle/data/{language}/{type}-articles-subtask-{subtask}/"
    )


def get_paths(language, model_name="default_model", subtask=3):
    """Function that constructs the train and dev paths for the given language

    Args:
    ------
        language (str): Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)
        model_name (str): Name of the model
        subtask (int): Subtask of the dataset (default is 3)

    Returns:
    --------
        paths (dict): Dictionary with paths to the train folder, dev folder, train labels file and dev labels file
    """
    paths = {
        "train_folder": get_articles_folder_path(language, "train", subtask),
        "dev_folder": get_articles_folder_path(language, "dev", subtask),
        "test_folder": get_articles_folder_path(language, "test", subtask),
        "train_labels_folder": get_labels_folder_path(
            language, "train", subtask
        ),
        "dev_labels_folder": get_labels_folder_path(language, "dev", subtask),
        "train_labels": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/train-labels-subtask-{subtask}.txt",
        ),
        "dev_labels": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/dev-labels-subtask-{subtask}.txt",
        ),
        "test_labels": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/test-labels-subtask-{subtask}.txt",
        ),
        "train_template": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/train-labels-subtask-{subtask}.template",
        ),
        "dev_template": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/dev-labels-subtask-{subtask}.template",
        ),
        "test_template": os.path.join(
            BASE_PATH,
            f"bundle/data/{language}/test-labels-subtask-{subtask}.template",
        ),
        "train_template_translated": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/train_st{subtask}_translated.txt",
        ),
        "dev_template_translated": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/dev_st{subtask}_translated.txt",
        ),
        "test_template_translated": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/test_st{subtask}_translated.txt",
        ),
        "train_predictions": os.path.join(
            BASE_PATH, f"outputs/{model_name}/{language}-train-predictions.txt"
        ),
        "train_metrics": os.path.join(
            BASE_PATH, f"outputs/{model_name}/{language}-train-metrics.txt"
        ),
        "dev_predictions": os.path.join(
            BASE_PATH, f"outputs/{model_name}/{language}-dev-predictions.txt"
        ),
        "dev_metrics": os.path.join(
            BASE_PATH, f"outputs/{model_name}/{language}-dev-metrics.txt"
        ),
        "train_embeddings": os.path.join(
            BASE_PATH,
            f"precomputed_embeddings/{model_name}/{language}/train_embeddings.pkl",
        ),
        "dev_embeddings": os.path.join(
            BASE_PATH,
            f"precomputed_embeddings/{model_name}/{language}/dev_embeddings.pkl",
        ),
        "test_embeddings": os.path.join(
            BASE_PATH,
            f"precomputed_embeddings/{model_name}/{language}/test_embeddings.pkl",
        ),
        "train_translated_template": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/train_st3_translated.txt",
        ),
        "dev_translated_template": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/dev_st3_translated.txt",
        ),
        "test_translated_template": os.path.join(
            BASE_PATH,
            f"data_translated/{language}/test_st3_translated.txt",
        ),
    }
    return paths


def get_logger(logger_name):
    """Function that creates a logger object and sets the logging level to INFO

    Args:
    ------
        logger_name (str): Name of the logger

    Returns:
    --------
        logger (logging.Logger): Logger object
    """

    os.makedirs(f"logger/", exist_ok=True)

    logger = logging.getLogger(logger_name)
    log_file = f"logger/{logger_name}.log"

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        f_handler = logging.FileHandler(log_file, mode="w")
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger
