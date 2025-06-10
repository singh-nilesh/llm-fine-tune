from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_tokenizer import DataTokenizer
from src.components.model_loader import ModelLoader
from src.components.trainer import Trainer
import torch
import gc
import os

def main():
    model = None
    try:
        # Set environment variable for better CUDA OOM handling
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logging.info("Starting data ingestion")
        try:
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.init_ingestion()
        except Exception as e:
            logging.error(f"Data ingestion failed: {str(e)}")
            raise CustomException("Failed during data ingestion stage", e)
        
        logging.info("Starting data tokenization")
        try:
            data_tokenizer = DataTokenizer(
                train_path=train_data_path,
                test_path=test_data_path,
                model_name=base_model_name
            )
            tokenized_dataset = data_tokenizer.init_tokenizer()
        except Exception as e:
            logging.error(f"Data tokenization failed: {str(e)}")
            raise CustomException("Failed during data tokenization stage", e)
        
        logging.info(f"Loading base model: {base_model_name}")
        try:
            loader = ModelLoader(base_model_name)
            model = loader.loader_init()
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise CustomException("Failed during model loading stage", e)
        
        logging.info("Starting training process")
        try:
            tuner = Trainer(
                model=model,
                tokenizer=data_tokenizer.tokenizer
            )
            tuner.train(tokenized_dataset)
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"Training process failed: {str(e)}")
            raise CustomException("Failed during training stage", e)
            
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise CustomException(e)
    finally:
        # Clean up GPU memory properly
        if model is not None:
            if hasattr(model, 'cpu'):
                model.cpu()  # Move model to CPU first
            del model
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force Python garbage collection
        gc.collect()
        logging.info("Memory cleanup completed")

if __name__ == "__main__":
    main()
