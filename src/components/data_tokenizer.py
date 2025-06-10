"""
This file tokenizes raw data for model inputs
"""
import json
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets import DatasetDict, Dataset, load_from_disk
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTokenizerConfig:
    artifacts_dir: str = os.environ.get('ARTIFACTS_DIR', os.path.join(os.path.dirname(__file__), 'artifacts'))
    
    @property
    def save_tokenizer_path(self):
        path = os.path.join(self.artifacts_dir, "tokenized")
        logging.info(f"Tokenized data path: {path}")
        return path

class DataTokenizer:
    def __init__(self, train_path, test_path, model_name):
        self.tokenizer_config = DataTokenizerConfig()
        self.train_data_path = train_path
        self.test_data_path = test_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    
    def init_tokenizer(self):
        """Initialize tokenizer by loading cached data or tokenizing raw data"""
        tokenized_path = self.tokenizer_config.save_tokenizer_path
        
        # Ensure tokenized directory exists
        os.makedirs(tokenized_path, exist_ok=True)
        logging.info(f"Checking for tokenized data at: {tokenized_path}")
        
        # Try to load existing tokenized data
        if self._has_cached_data(tokenized_path):
            cached_data = self._load_cached_data(tokenized_path)
            if cached_data is not None:
                return cached_data
        
        # No cached data available, tokenize from scratch
        logging.info("Tokenizing raw data from scratch")
        raw_data = self.load_raw_data()
        return self.tokenize_and_save(raw_data)
    
    
    def _has_cached_data(self, path):
        """Check if valid cached tokenized data exists"""
        if not os.path.exists(path):
            return False
        
        files = os.listdir(path)
        has_arrow_files = any(f.endswith('.arrow') for f in files)
        has_dataset_files = any(f in ['dataset_info.json', 'state.json'] for f in files)
        
        return bool(files) and (has_arrow_files or has_dataset_files)
    
    
    def _load_cached_data(self, path):
        """Attempt to load cached tokenized data"""
        try:
            logging.info("Loading cached tokenized data from disk")
            return load_from_disk(path)
        except Exception as e:
            logging.warning(f"Failed to load cached data: {e}")
            logging.info("Will tokenize raw data instead")
            return None
    
    def func_tokenize(self, examples):
        """Process examples in batches, ensuring all inputs are properly cast to strings"""
        # For batched processing, handle lists of inputs
        if isinstance(examples['instruction'], list):
            instructions = [str(inst) if inst is not None else "" for inst in examples['instruction']]
            responses = [str(resp) if resp is not None else "" for resp in examples['response']]
        else:
            # Handle single example case
            instructions = [str(examples['instruction']) if examples['instruction'] is not None else ""]
            responses = [str(examples['response']) if examples['response'] is not None else ""]
        
        # Tokenize instructions and responses separately
        tokenized_inputs = self.tokenizer(
            instructions,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        tokenized_targets = self.tokenizer(
            responses,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        # Return the tokenized data with labels
        tokenized_inputs['labels'] = tokenized_targets['input_ids']
        return tokenized_inputs
    
    
    def load_jsonl_file(self, file_path):
        """Load JSONL file directly without using load_dataset"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON on line {line_num} in {file_path}: {e}")
            logging.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            raise CustomException(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise CustomException(f"Error loading file {file_path}: {e}")
    
    
    def load_raw_data(self):
        """Load raw data using direct file reading instead of load_dataset"""
        logging.info(f"Loading raw data from: {self.train_data_path} and {self.test_data_path}")
        
        try:
            # Load data directly from JSONL files
            train_data_list = self.load_jsonl_file(self.train_data_path)
            test_data_list = self.load_jsonl_file(self.test_data_path)
            
            # Create datasets from the loaded data
            train_dataset = Dataset.from_list(train_data_list)
            test_dataset = Dataset.from_list(test_data_list)
            
            logging.info("Raw data loaded for Tokenization")
            return DatasetDict({"train": train_dataset, "test": test_dataset})
            
        except Exception as e:
            logging.error(f"Failed to load raw data: {e}")
            # Fallback: try using pandas
            logging.info("Attempting to load data using pandas as fallback")
            try:
                train_df = pd.read_json(self.train_data_path, lines=True)
                test_df = pd.read_json(self.test_data_path, lines=True)
                
                train_dataset = Dataset.from_pandas(train_df)
                test_dataset = Dataset.from_pandas(test_df)
                
                logging.info("Raw data loaded using pandas fallback")
                return DatasetDict({"train": train_dataset, "test": test_dataset})
            except Exception as fallback_error:
                logging.error(f"Fallback method also failed: {fallback_error}")
                raise CustomException(f"Failed to load data with both methods: {e}, {fallback_error}")
    
    
    def tokenize_and_save(self, raw_data_dict: DatasetDict):
        """Tokenize and save data with better error handling"""
        try:
            logging.info("Tokenizing raw data...")
            tokenized_data = raw_data_dict.map(
                self.func_tokenize, 
                batched=True,
                remove_columns=raw_data_dict["train"].column_names  # Remove original columns
            )
            
            logging.info(f"Saving tokenized data to: {self.tokenizer_config.save_tokenizer_path}")
            
            # Clean up the save directory first
            if os.path.exists(self.tokenizer_config.save_tokenizer_path):
                import shutil
                shutil.rmtree(self.tokenizer_config.save_tokenizer_path)
            
            os.makedirs(self.tokenizer_config.save_tokenizer_path, exist_ok=True)
            tokenized_data.save_to_disk(self.tokenizer_config.save_tokenizer_path)
            
            logging.info("Finished Tokenizing the raw data")
            return tokenized_data
            
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            raise CustomException(f"Tokenization failed: {e}")

if __name__ == "__main__":
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    train_path = 'artifacts/train.jsonl'  
    test_path = 'artifacts/test.jsonl'
    tokenizer = DataTokenizer(train_path, test_path, model_id)
    toke_data = tokenizer.init_tokenizer()