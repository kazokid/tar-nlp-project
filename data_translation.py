from tqdm import tqdm
import csv
import bundle.baselines.st3 as st3
import bundle.baselines.st2 as st2
from models.helpers import *
import pandas as pd
from easynmt import EasyNMT


# def make_dataframe(input_folder):
#     # MAKE TXT DATAFRAME
#     text = []

#     for fil in tqdm(
#         filter(lambda x: x.endswith(".txt"), os.listdir(input_folder))
#     ):

#         iD, txt = (
#             fil[7:].split(".")[0],
#             open(input_folder + fil, "r", encoding="utf-8").read(),
#         )
#         text.append((iD, txt))

#     df_text = pd.DataFrame(text, columns=["id", "text"])
#     df = df_text

#     return df


def row_translator_st3(row, model: EasyNMT, logger, lang):
    try:
        # TODO this might raise an exception because of the auto detection of the language

        print(row["text"])
        translation = model.translate(
            row["text"],
            target_lang="en",
            source_lang="fr",
            show_progress_bar=True,
        )

        if translation is not None:
            return pd.Series(
                {
                    "id": row["id"],
                    "line": row["line"],
                    "text": translation,
                }
            )
        else:
            logger.error(
                f"Error translating article id:{row.id} row: {row.line}: Translation is None"
            )
            return None

    except Exception as e:
        logger.error(
            f"Error translating article id:{row.id} row: {row.line}: {e}"
        )
        logger.error(
            f"\nThis is the text:{row.text}\nThis is the translated text: {translation}"
        )

        return None


# def row_translator_st2(row, translator, logger):
#     """TODO:ignore"""
#     translation = None

#     try:

#         translator.raise_Exception = True
#         translation = translator.translate(row["text"], dest="en")

#         if translation is not None:
#             return pd.Series(
#                 {
#                     "id": row["id"],
#                     "text": translation.text,
#                 }
#             )
#         else:
#             logger.error(
#                 f"Error translating article id:{row.id}\nThis is the text: {row.text}\nThis is the translation: {translation}"
#             )
#             return None

#     except Exception as e:
#         # TODO implement a retry mechanism
#         logger.error(
#             f"Error translating article id:{row.id}\nThis is the text: {row.text}\nThis is the translation: {translation}\nERROR:{e}"
#         )
#         return None


languages_train = ["fr", "ge", "it", "po", "ru"]
languages_only_test = ["es", "gr", "ka"]
all_languages = languages_train + languages_only_test

data_type = ["train", "dev", "test"]

# subtask 3
for lang in all_languages:
    for type in data_type:
        logger = get_logger(f"st3_{lang}_{type}_translation")
        if lang in languages_only_test and type != "test":
            continue

        paths_st3 = get_paths(lang)

        text = []

        with open(
            paths_st3[f"{type}_template"],
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
        print(df)
        df.id = df.id.apply(int)
        df.line = df.line.apply(int)
        df = df[df.text.str.strip().str.len() > 0].copy()

        os.makedirs(f"data_translated/{lang}/", exist_ok=True)

        if os.path.exists(f"data_translated/{lang}/{type}_st3_translated.txt"):
            os.remove(f"data_translated/{lang}/{type}_st3_translated.txt")

        # Translating the subtask 3 data in batches of 100

        for i in range(0, len(df), 100):
            batch = df.iloc[i : i + 100]
            # batch = df.iloc[0]

            model = EasyNMT("m2m_100_418M", cache_folder=BASE_PATH)

            df_translated = batch.apply(
                lambda row: row_translator_st3(row, model, logger),
                axis=1,
            )
            # i tried to use row translate on a single row but it still did not work
            # df_translated = row_translator_st3(batch, model, logger, lang)

            with open(
                f"data_translated/{lang}/{type}_st3_translated.txt", "a"
            ) as file:
                df_translated.to_csv(
                    file,
                    index=False,
                    header=False,
                    quoting=csv.QUOTE_MINIMAL,
                    sep="\t",
                )

            logger.info(f"Translated {i + 100} out of {df.shape[0]} lines")


# subtask 2; ignore this, we don't need it for now

# for lang in all_languages:
#     for type in data_type:
#         logger = get_logger(f"st2_{lang}_{type}_translation")
#         if lang in languages_only_test:
#             if type != "test":
#                 continue

#         paths_st2 = get_paths(lang)

#         df = make_dataframe(paths_st2[f"{type}_folder"])

#         print(df)

#         folder = f"data_translated/{lang}/{type}-articles-subtask-2"
#         os.makedirs(folder, exist_ok=True)

#         for i in range(0, len(df)):
#             article = df.iloc[i]

#             translator = Translator()

#             print(f"the row is {article}")

#             translated_article = row_translator_st2(article, translator, logger)

#             if translated_article is not None:
#                 with open(f"{folder}/article{article.id}.txt", "w") as file:
#                     text = translated_article.text
#                     file.write(text)

#                 logger.info(f"Translated {i} out of {df.shape[0]} articles")
#             else:
#                 logger.error(f"Error translating article id:{article.id}")
