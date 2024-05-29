import pandas as pd
import sys
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory


from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
from bundle.scorers.scorer_subtask_3 import *
import bundle.baselines.st3 as bundle_baseline
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function
from models.dataset_preparation import *

import pickle


def main():
    paths: dict = get_paths("en", "precomputed_all_roberta_large")

    folder_dev = paths["dev_folder"]
    out_fn = paths["dev_predictions"]

    # Read Data
    print("Loading dataset...")
    test = bundle_baseline.make_dataframe(folder_dev)

    with open("labels_en_train.pkl", "rb") as f:
        Y_train = pickle.load(f)

    X_test = test["text"].values

    multibin = MultiLabelBinarizer()  # use sklearn binarizer

    Y_train = multibin.fit_transform(Y_train)
    # Create train-test split

    # load train embeddings
    with open("embeddings_en_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # load test embeddings
    with open("embeddings_en_dev.pkl", "rb") as f:
        X_test = pickle.load(f)

    print("Fitting logistic regression...")
    model = MultiOutputClassifier(
        LogisticRegression(tol=1e-4, max_iter=int(1e7), n_jobs=-1, verbose=1)
    )
    model.fit(X_train, Y_train)

    print("In-sample Acc: \t\t", model.score(X_train, Y_train))

    Y_pred = model.predict(X_test)
    out = multibin.inverse_transform(Y_pred)
    out = list(map(lambda x: ",".join(x), out))
    out = pd.DataFrame(out, test.index)
    out.to_csv(out_fn, sep="\t", header=None)
    print("Results on: ", out_fn)


if __name__ == "__main__":
    main()
