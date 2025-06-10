
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_tokenizer import DataTokenizer
from src.components.model_loader import ModelLoader
from src.components.trainer import Trainer

def main():
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Pre-processing
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.init_ingestion()
    
    # Tokenize the data
    data_tokenizer = DataTokenizer(
        train_path= train_data_path,
        test_path= test_data_path,
        model_name= base_model_name
    )
    tokenized_dataset = data_tokenizer.init_tokenizer()
    
    # Load model
    loader = ModelLoader(base_model_name)
    model = loader.loader_init()
    
    # trainer
    tuner = Trainer(
        model=model,
        tokenizer= data_tokenizer.tokenizer
    )
    tuner.train(tokenized_dataset)
    