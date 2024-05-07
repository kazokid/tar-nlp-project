import ast
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        model_name,
        max_length=200,
        text_column="text",
        label_column="labels",
    ):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.one_hot = MultiLabelBinarizer()

        # Apply one-hot encoding to labels
        transf = self.one_hot.fit_transform(self.data[label_column].values)
        self.labels_encoded = self.one_hot.fit_transform(
            self.data[label_column].fillna("").str.split(",").values
        ).tolist()  # remove the first column -it typically marks beginning of the sentece
        # MultiLabelBinarizer automatically determines the number of unique labels

    def safe_literal_eval(self, x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            print("Error evaluating:", x)
            return []

    def multi_label_binarizer(self):
        return self.one_hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Function to retrieve the data sample at the specified index.

        Args:
        ------
            idx (int): The index of the data sample.

        Returns:
        --------
            input_ids (torch.Tensor): Tensor containing the IDs of each token in the sentence. Each ID represents the token's index in the vocabulary of the pretrained language model.
            attention_mask (torch.Tensor): Tensor containing the attention mask for the input_ids. It masks padding tokens, ensuring that the model does not attend to them during training or inference.
            labels (torch.Tensor): Tensor containing the predicted one-hot encoded labels.
        """
        text = self.data.iloc[idx][self.text_column]
        labels = self.labels_encoded[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding[
            "input_ids"
        ].squeeze()  # Squeeze to remove extra dimensions
        attention_mask = encoding["attention_mask"].squeeze()

        labels = torch.tensor(labels, dtype=torch.float32)

        return input_ids, attention_mask, labels


# vidjeti kako odrediti optimalan max_length ?
def create_dataloader(data, model_name, batch_size=16, max_length=200):

    dataset = CustomDataset(data, model_name, max_length)
    one_hot_encoder = dataset.multi_label_binarizer()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, one_hot_encoder
