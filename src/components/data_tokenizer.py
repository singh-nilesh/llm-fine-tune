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
from config import Config

class DataTokenizer:
    def __init__(self):
        self.config = Config()
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """ Load tokenizer and add special tokens"""
        try:
            # try to load tokenizer from disk
            if os.path.exists(self.config.tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                logging.info(f"Finished loading tokenizer from {self.config.tokenizer_path}")
            
            # new tokenizer
            else:
                logging.info("Tokenizer not found Loading new Tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                special_tokens = {
                    "additional_special_tokens" : [
                        "<|user|>", "<|assistant|>", "<|end|>",
                        "<bos>", "<start_of_turn>", "<end_of_turn>",
                        ]
                }
                tokenizer.add_special_tokens(special_tokens)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                
                try:
                    # make sure the dir exists
                    os.makedirs(os.path.dirname(self.config.tokenizer_path), exist_ok=True)
                except OSError as e:
                    raise CustomException("Failed to create Tokenizer dir", sys)
                
                # save tokenizer to disk
                tokenizer.save_pretrained(self.config.tokenizer_path)
                logging.info("Tokenizer loaded and special tokens added, and saved to disc")
            return tokenizer
        except Exception as e:
            raise CustomException("Error loading tokenizer", sys)


    def load_jsonl(self, file_path):
        """Load .jsonl data from disk - Method: Manual loading"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data


    def load_data(self):
        """Loads raw data, returns DatasetDict with train and eval splits"""
        try:
            train_data = self.load_jsonl(self.config.train_data_path)
            eval_data = self.load_jsonl(self.config.eval_data_path)
            
            dataset = DatasetDict({
                "train": Dataset.from_list(train_data),
                "test": Dataset.from_list(eval_data)
            })
            logging.info("Dataset loaded manually using DatasetDict")
            return dataset
        except Exception as e:
            raise CustomException(f"Error loading dataset: {e}", sys)


    def func_tokenize(self, example):
        """This function is used as template for tokenizing each line of data"""
        try:
            # validation for required key
            if 'text' not in example:
                raise ValueError("Example missing required 'text' key")

            return self.tokenizer(
                example['text'],
                truncation=True,
                max_length=self.config.max_length,
                padding=False  # Handled dynamically by collator
            )
        except Exception as e:
            raise CustomException(f"Error during raw data tokenization: {e}", sys)
    
    
    def get_tokenized_data(self):
        """tokenize raw data from disk"""
        try:
            raw_data = self.load_data()
            tokenized_data = raw_data.map(self.func_tokenize, batched=False)
            
            try:
                # Make sure the directory exists
                os.makedirs(os.path.dirname(self.config.tokenize_data_path), exist_ok=True)
            except OSError as e:
                raise CustomException("Failed to create tokenizer_data dir", sys)
            
            # save tokenized data to disk
            tokenized_data.save_to_disk(self.config.tokenize_data_path)
            logging.info("Finished tokenizing raw data")
            return tokenized_data
        except Exception as e:
            raise CustomException(f"Error during tokenizer - data mapping: {e}", sys)

if __name__ == "__main__":
    data_tokenizer = DataTokenizer()
    tokenized_dataset = data_tokenizer.get_tokenized_data()
    print(f"Tokenized dataset saved to {data_tokenizer.config.tokenize_data_path}")