"""
    Parse Supabase records to , Q&A pairs (.json) for model
"""
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import json
from supabase import create_client, Client


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.environ.get('ARTIFACTS_DIR', os.path.join(os.path.dirname(__file__), 'artifacts'))
    train_data_path: str = os.path.join(artifacts_dir, 'train.jsonl')
    test_data_path: str = os.path.join(artifacts_dir, 'test.jsonl')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.SUPABASE_URL= os.getenv('SUPABASE_URL')
        self.SUPABASE_KEY= os.getenv('SUPABASE_KEY')
    
    def init_ingestion(self):
        # Supabase init
        supabase = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        
        # retrive all records
        df = self.fetch_all_rows_paginated(supabase)
        
        # train-test split
        train_df,test_df = train_test_split(df, test_size=0.1, random_state=42)
        logging.info("split data into train-test set")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
        
        # write to JSON
        self.write_to_json(train_df, self.ingestion_config.train_data_path)
        self.write_to_json(test_df, self.ingestion_config.test_data_path)
        
        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
        
        
    def write_to_json(self, df:pd.DataFrame, filename:str):         
        """
        this function saves Df to .jsonl format
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                try:
                    f.write(json.dumps({
                        "instruction": row["prompt"],
                        "response": json.loads(row["response"])  # assumes response is a JSON string
                    }) + "\n")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse response at row {_}: {e}")
        logging.info(f"Saved {filename} with {len(df)} records.")
        
        
    def fetch_all_rows_paginated(self, supabase, table="roadmap", page_size=1000):
        """
        this function retrieves, records from Supabase, using pagination (supabase limit)
        """
        all_data = []
        offset = 0

        while True:
            try:
                res = supabase.table(table).select("*").order("id").range(offset, offset + page_size - 1).execute()
            except Exception as e:
                raise CustomException(f"supabase query error: {e}", sys)
                
            batch = res.data
            if not batch:
                break
            all_data.extend(batch)
            logging.info(f"Fetched rows {offset} to {offset + len(batch) - 1}")
            if len(batch) < page_size:
                break
            offset += page_size
            
        logging.info("finished retrieving records from Supabase")
        return pd.DataFrame(all_data)



if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    data_files = ingestion_obj.init_ingestion()
    print(data_files)