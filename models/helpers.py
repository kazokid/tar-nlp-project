import os

BASE_PATH = os.getenv("MY_REPO_LOCATION")
CLASSES_SUBTASK_3_PATH = os.path.join(
    BASE_PATH, "bundle/scorers/techniques_subtask3.txt"
)


def get_articles_folder_path(language, type):
    """Function that constructs the path to the folder given language and type (train, dev, test)

    Args:
    ------
        language (str): Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)
        type (str): Type of the dataset (train, dev, test)

    Returns:
    --------
        path (str): Path to the folder with the articles of the given language and type
    """

    return os.path.join(BASE_PATH, f"bundle/data/{language}/{type}-articles-subtask-3/")


def get_paths(language):
    """Function that constructs the train and dev paths for the given language

    Args:
    ------
        language (str): Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)

    Returns:
    --------
        paths (dict): Dictionary with paths to the train folder, dev folder, train labels file and dev labels file
    """
    paths = {
        "train_folder": get_articles_folder_path(language, "train"),
        "dev_folder": get_articles_folder_path(language, "dev"),
        "train_labels": os.path.join(
            BASE_PATH, f"bundle/data/{language}/train-labels-subtask-3.txt"
        ),
        "dev_labels": os.path.join(
            BASE_PATH, f"bundle/data/{language}/dev-labels-subtask-3.txt"
        ),
        "train_predictions": os.path.join(
            BASE_PATH, f"outputs/bert_baseline/{language}-train-predictions.txt"
        ),
        "train_metrics": os.path.join(
            BASE_PATH, f"outputs/bert_baseline/{language}-train-metrics.txt"
        ),
        "dev_predictions": os.path.join(
            BASE_PATH, f"outputs/bert_baseline/{language}-dev-predictions.txt"
        ),
        "dev_metrics": os.path.join(
            BASE_PATH, f"outputs/bert_baseline/{language}-dev-metrics.txt"
        ),
    }
    return paths
