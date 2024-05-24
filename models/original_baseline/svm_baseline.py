import pandas as pd
from tqdm import tqdm
import os
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

BASE_PATH = (
    __file__.replace("\\", "/").split("tar-nlp-project")[0] + "tar-nlp-project/"
)

sys.path.append(BASE_PATH)  # enable importing from the root directory

from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
import bundle.scorers.scorer_subtask_3 as bundle_scorer
import bundle.baselines.st3 as bundle_baseline
import models.helpers as helpers
import models.dataset_preparation as dataset_preparation


def main():

    # TODO check and finish to run the class model
    parser = argparse.ArgumentParser(description="Subtask-3")
    parser.add_argument(
        "language",
        type=str,
        nargs=1,
        help="Language of the dataset (en, es, fr, ge, gr, it, ka, po, ru)",
    )

    args = parser.parse_args()
    language = args.language[0]

    paths: dict = helpers.get_paths(language, "original_baseline")

    train = bundle_baseline.make_dataframe(
        paths["train_folder"], paths["train_labels"]
    )
    test = bundle_baseline.make_dataframe(paths["dev_folder"])

    X_train = train["text"].values
    Y_train = train["labels"].fillna("").str.split(",").values

    X_test = test["text"].values

    # print the number of rows in X_test

    print("Number of rows in X_test: ", len(X_test))
    print("location of test file is : ", paths["dev_folder"])

    multibin = MultiLabelBinarizer()  # use sklearn binarizer

    Y_train = multibin.fit_transform(Y_train)
    # Create train-test split

    pipe = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(ngram_range=(1, 2), analyzer="word"),
            ),
            (
                "SVM_multiclass",
                MultiOutputClassifier(
                    svm.SVC(class_weight=None, C=1, kernel="linear"), n_jobs=1
                ),
            ),
        ]
    )

    print("Fitting SVM...")
    pipe.fit(X_train, Y_train)

    print("In-sample Acc: \t\t", pipe.score(X_train, Y_train))

    Y_pred = pipe.predict(X_test)
    out = multibin.inverse_transform(Y_pred)
    out = list(map(lambda x: ",".join(x), out))
    out = pd.DataFrame(out, test.index)
    out.to_csv(paths["dev_predictions"], sep="\t", header=None)
    print("Results on: ", paths["dev_predictions"])


if __name__ == "__main__":
    main()
