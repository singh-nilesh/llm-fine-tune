"""
This file tokenizes raw data for model inputs, including attention masks.
"""
import json
import os
import sys
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from src.exception import CustomException
from src.logger import logging
from src.components.config import Config

class DataTokenizer:
    def __init__(self):
        self.config = Config()

    def tokenizer_init(self):
        """ Load tokenizer and add special tokens"""
        try:
            # try to load tokenizer from disk - check if it's a valid tokenizer directory
            if os.path.exists(os.path.join(self.config.tokenizer_path, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                logging.info(f"Finished loading tokenizer from {self.config.tokenizer_path}")
            else:
                logging.info(f"Tokenizer not found at {self.config.tokenizer_path}. Loading new tokenizer from {self.config.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                special_tokens = {
                    "additional_special_tokens": [
                        "<|user|>", "<|assistant|>", "<|end|>",
                        "<bos>", "<start_of_turn>", "<end_of_turn>",
                    ]
                }
                tokenizer.add_special_tokens(special_tokens)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                try:
                    os.makedirs(os.path.dirname(self.config.tokenizer_path), exist_ok=True)
                except OSError as e:
                    raise CustomException(f"Failed to create Tokenizer dir: {e}", sys)
                tokenizer.save_pretrained(self.config.tokenizer_path)
                logging.info("Tokenizer loaded and special tokens added, and saved to disc")
            return tokenizer
        except Exception as e:
            raise CustomException(f"Error loading tokenizer: {e}", sys)

    def load_jsonl(self, file_path):
        """Load .jsonl data from disk - Method: Manual loading"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append(json.loads(line))
            return data
        except Exception as e:
            raise CustomException(f"Error loading JSONL file {file_path}: {e}", sys)

    def load_data(self):
        """Loads raw data, returns DatasetDict with train and eval splits. Each example must have a 'text' field."""
        try:
            train_data = self.load_jsonl(self.config.train_data_path)
            eval_data = self.load_jsonl(self.config.eval_data_path)
            dataset = DatasetDict({
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data)
            })
            logging.info("Dataset loaded manually using DatasetDict (raw text, not tokenized)")
            return dataset
        except Exception as e:
            raise CustomException(f"Error loading dataset: {e}", sys)

if __name__ == "__main__":
    tokenizer_inst = DataTokenizer()
    tokenizer = tokenizer_inst.tokenizer_init()
    raw_dataset = tokenizer_inst.load_data()
    print(raw_dataset)  # For debug: show structure of raw dataset