"""
    this file tokenizes raw data for model inputs
"""
import json
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets import DatasetDict, load_dataset, load_from_disk
import os

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTokenizerConfig:
    save_tokenizer_path =  os.path.join("artifacts", "tokenized")

class DataTokenizer:
    def __init__(self, train_path, test_path, model_name):
        self.tokenizer_config = DataTokenizerConfig()
        self.train_data_path = os.path.join(os.path.dirname(__file__), train_path)
        self.test_data_path = os.path.join(os.path.dirname(__file__), test_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def init_tokenizer(self):
        # load from disk if available
        if os.path.exists(self.tokenizer_config.save_tokenizer_path):
            logging.info("loading Tokenized data from disk")
            return load_from_disk(self.tokenizer_config.save_tokenizer_path)
        
        raw_data = self.load_raw_data()
        return self.tokenize_and_save(raw_data)
    
    
    def func_tokenize(self, examples):
        """
        Process examples in batches, ensuring all inputs are properly cast to strings
        """
        # For batched processing, handle lists of inputs
        if isinstance(examples['instruction'], list):
            instructions = [str(inst) if inst is not None else "" for inst in examples['instruction']]
            responses = [str(resp) if resp is not None else "" for resp in examples['response']]
        else:
            # Handle single example case
            instructions = str(examples['instruction']) if examples['instruction'] is not None else ""
            responses = str(examples['response']) if examples['response'] is not None else ""
        
        return self.tokenizer(
            instructions,
            text_target=responses,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        
    def load_raw_data(self):
        train_data = load_dataset("json", data_files=self.train_data_path, split='train')
        test_data = load_dataset('json', data_files=self.test_data_path, split='train')
        logging.info("raw data loaded for Tokenization")
        return DatasetDict({"train": train_data, "test": test_data})
    
    def tokenize_and_save(self, raw_data_dict: DatasetDict):
        tokenized_data = raw_data_dict.map(self.func_tokenize, batched=True)
        tokenized_data.save_to_disk(self.tokenizer_config.save_tokenizer_path)
        logging.info("Finished Tokenizing the raw data")
        return tokenized_data


if __name__ == "__main__":
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    train_path = 'artifacts/train.jsonl'
    test_path = 'artifacts/test.jsonl'
    
    tokenizer = DataTokenizer(train_path, test_path, model_id)
    toke_data = tokenizer.init_tokenizer()