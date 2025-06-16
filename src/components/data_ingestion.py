"""
Parse Supabase records to instruction-following Q&A pairs in .jsonl format for model fine-tuning.
"""
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from supabase import create_client, Client
from src.components.config import Config

class DataIngestion:
    def __init__(self):
        self.Config = Config()
        self.SUPABASE_URL = os.getenv('SUPABASE_URL')
        if not self.SUPABASE_URL:
            raise CustomException("SUPABASE_URL environment variable not set", sys)
        
        self.SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        if not self.SUPABASE_KEY:
            raise CustomException("SUPABASE_KEY environment variable not set", sys)


    def init_ingestion(self):
        """main controler"""
        supabase = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        df = self.fetch_all_rows_paginated(supabase)

        train_df, test_df = train_test_split(df, test_size=self.Config.test_size, random_state=42)
        logging.info("Split data into train-test set")

        os.makedirs(os.path.dirname(self.Config.train_data_path), exist_ok=True)

        self.write_to_jsonl(train_df, self.Config.train_data_path)
        self.write_to_jsonl(test_df, self.Config.eval_data_path)
        logging.info("Data successfully exported to .jsonl format")

        return (
            self.Config.train_data_path,
            self.Config.eval_data_path
        )


    def format_text(self, prompt: str, response: str, model_type: str = "default") -> str:
        """Format instruction and response into a structured prompt-completion string."""
        if model_type == "gemma":
            return (
                f"<bos><start_of_turn>user {prompt}<end_of_turn>"
                f"<start_of_turn>model {response}<end_of_turn>"
            )
        else:
            return (
                f"<|user|>\n{prompt}\n<|end|>\n"
                f"<|assistant|>\n{response}<|end|>"
            )


    def write_to_jsonl(self, df: pd.DataFrame, filename: str):
        """Convert each row into formatted text field and save as .jsonl"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            for idx, row in df.iterrows():
                try:
                    formatted_text = self.format_text(row['prompt'], row['response'], model_type="default")
                    f.write(json.dumps({"text": formatted_text}) + "\n")
                except Exception as e:
                    logging.error(f"Failed to process row {idx}: {e}")
        logging.info(f"Saved {filename} with {len(df)} records.")


    def fetch_all_rows_paginated(self, supabase, table="roadmap", page_size=1000):
        """Retrieve records from Supabase using pagination"""
        all_data = []
        offset = 0

        while True:
            try:
                res = supabase.table(table).select("*").order("id").range(offset, offset + page_size - 1).execute()
            except Exception as e:
                raise CustomException(f"Supabase query error: {e}", sys)

            batch = res.data
            if not batch:
                break
            all_data.extend(batch)
            logging.info(f"Fetched rows {offset} to {offset + len(batch) - 1}")
            if len(batch) < page_size:
                break
            offset += page_size

        logging.info("Finished retrieving records from Supabase")
        return pd.DataFrame(all_data)

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    data_files = ingestion_obj.init_ingestion()
    print(data_files)
