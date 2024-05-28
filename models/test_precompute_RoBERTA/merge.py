import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import pickle
import os
import sys
from tqdm import tqdm

sys.path.append(
    os.getenv("MY_REPO_LOCATION")
)  # enable importing from the root directory


from bundle.scorers.scorer_subtask_3 import _read_csv_input_file
from bundle.scorers.scorer_subtask_3 import *
import bundle.baselines.st3 as bundle_baseline
from models.helpers import *  # myb switch to helper. notation use to make it clear the source of the function
from models.dataset_preparation import *
import pickle

def main():

    
    paths: dict = get_paths("en", "precomputed_all_roberta_large")

    dev_articles_fn = paths["dev_folder"]

    # Read Data
    print("Loading dataset...")
    dev_articles_l = bundle_baseline.make_dataframe(dev_articles_fn)
    print(dev_articles_l.shape)


    try:
        with open("embeddings_en_dev.pkl", "rb") as f:
            X_dev = pickle.load(f)

        rows = []

        # Iterate over both X_dev and dev_articles_l
        for (article_id, line), (_, row), embedding in zip(dev_articles_l.index, dev_articles_l.iterrows(), X_dev):

            # Create a new row with the article_id, line, and embedding columns
            new_row = {'id': article_id, 'line': line}
            new_row.update({f'embedding_{i}': val for i, val in enumerate(embedding)})
            rows.append(new_row)
        df_concatenated = pd.DataFrame(rows)

        # Set the index of the merged DataFrame
        df_concatenated = df_concatenated.set_index(['id', 'line'])
    
        print(df_concatenated.index)
        print(df_concatenated.iloc[0])
       

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()