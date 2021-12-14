from random import random

import random
import transformers
import pandas as pd
from datasets import ClassLabel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

print(transformers.__version__)


class MaskedLanguageModeling:
    def __init__(self,
                 dataset,
                 checkpoint,
                 evaluation_strategy="epoch",
                 learning_rate=2e-5,
                 weight_decay=0.01):
        self.datasets = dataset
        self.model_checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        self.tokenized_datasets = self.datasets.map(
            self._tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"]
        )
        self.lm_dataset = self.tokenized_datasets.map(
            self._group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )
        self.fine_tuning_model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
        self.training_args = TrainingArguments(
            "test-clm",
            evaluation_strategy,
            learning_rate,
            weight_decay,
        )
        self.trainer = Trainer(
            model=self.fine_tuning_model,
            args=self.training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["validation"],
        )

    def show_train_data(self):
        self._show_data(self.datasets["train"])

    def show_test_data(self):
        self._show_data(self.datasets["test"])

    def show_validation_data(self):
        self._show_data(self.datasets["validation"])

    def decode_text_for_language_model(self):
        print(self.tokenizer.decode(self.lm_dataset["train"][1]["input_ids"]))

    def train(self):
        self.trainer.train()

    def _group_texts(self, examples, block_size=128):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _show_data(self, data, num_examples=10):
        assert num_examples <= len(data), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(data) - 1)
            while pick in picks:
                pick = random.randint(0, len(data) - 1)
            picks.append(pick)

        df = pd.DataFrame(data[picks])
        for column, typ in data.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        print(df)

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"])
