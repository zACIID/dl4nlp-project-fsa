import torch
import datasets
import lightning as L
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# Initial reference:
# https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/text-transformers/text-transformers.py#L237
class TrainValDataModule(L.LightningDataModule): # TODO rename to FinBERTTrainVal data module
    def __init__(
            self,
            model_name_or_path: str = "ahmedrachid/FinancialBERT-Sentiment-Analysis",
            train_batch_size: int = 64,
            eval_batch_size: int = 8,
            train_split_size: float = 0.8,
            pin_memory: bool = False,
            prefetch_factor: int = 4,
            num_workers: int = 0,
            **kwargs,
    ):
        """
        :param model_name_or_path: huggingface model name or path, used to load the tokenizer
        :param train_batch_size:
        :param eval_batch_size: val/test/predict batch size
        :param train_split_size: fraction of data used for training.
            The remaining fraction of data will be used for validation
        :param kwargs:
        """

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

        self.tokenizer: transformers.PreTrainedTokenizerBase | None = None
        self.dataset: datasets.Dataset | None = None
        self.columns: datasets.Dataset | None = None

    def prepare_data(self):
        # This method is called at the beginning to basically load (i.e. read from file)/download the data
        # datasets.load_dataset("glue", self.task_name)
        datasets.load_dataset('glue', 'mrpc')
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str = None):
        # TODO should I ignore stage? because train and val datasets are created at the same moment

        # This method actually creates the dataset(s) and it is where feature-engineering should be
        #   (e.g. dropping columns, tokenizer calls, etc.)
        # TODO ideally here I should process all the datasets (which datasets? see docs/datasets-notes.md)
        #   and then split them via something similar to sklearn's train_test_split method
        # TODO 2: try to use spark to read datasets and perform operations on them since it allows for easy
        #   parallel computing so that every core of this machine can be leveraged
        #   Plain pandas is single core
        # TODO 3: I might want to process stuff via spark in another class, because the result of such pre-processing
        #   will be used by different dataloaders (finberttrainval, finberttest, HandEngTrainVal, HandEngTest, E2ETrainVal, E2ETest)

        # self.dataset = datasets.load_dataset("ElKulako/stocktwits-emoji")
        #
        # for split in self.dataset.keys():
        #     self.dataset[split] = self.dataset[split].map(
        #         self.convert_to_features,
        #         batched=True,
        #         remove_columns=["label"],
        #     )
        #     self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
        # https://huggingface.co/docs/datasets/v1.13.0/package_reference/main_classes.html#datasets.Dataset.set_format
        #     self.dataset[split].set_format(type="torch", columns=self.columns)
        #
        # self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        self.dataset = datasets.load_dataset('glue', 'mrpc')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        for split in ['train', 'validation']:
            # TODO here I must use padding='max_length' along with batched, else it doesn't work because batches will be padded to different lengths
            #   what I could do, which also avoids the parallelism warning, is to tokenize with fast tokenizer
            #   outside of the lambda: I pay the price of not-lazy loading but I ideally use less memory because batches
            #   anre't padded to max length and I can also use parallel dataloaders
            self.dataset[split] = self.dataset[split].map(lambda e: self.tokenizer(e['sentence1'], padding='max_length', return_tensors='pt', max_length=160), batched=True)

            # TODO i think setting type='torch' is redundant since tokenizer already returns pytorch tensors?
            self.dataset[split].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

            # self.dataset[split].remove_columns(column_names=['label'])

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        raise NotImplementedError("This data module is only for training and validation datasets")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("This data module is only for training and validation datasets")

    def convert_to_features(self, example_batch, indices=None):
        # # Either encode single sentence or sentence pairs
        # if len(self.text_fields) > 1:
        #     texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # else:
        #     texts_or_text_pairs = example_batch[self.text_fields[0]]
        #
        # # Tokenize the text/text pairs
        # features = self.tokenizer.batch_encode_plus(
        #     texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        # )
        #
        # # Rename label to labels to make it easier to pass to model forward
        # features["labels"] = example_batch["label"]
        #
        # return features
        # TODO maybe add pre-processing/engineering logic here
        pass
