import os
import sys
import gc
import torch

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_tokenizer import DataTokenizer
from src.components.model_loader import ModelLoader
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """Main training pipeline for model fine-tuning."""
    
    def __init__(self, base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self._setup_environment()
    
    
    def _setup_environment(self):
        """Set up environment variables and configurations."""
        # Set environment variable for better CUDA OOM handling
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Define artifacts directory
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
        os.environ["ARTIFACTS_DIR"] = artifacts_dir
        
        # Create artifacts directory if it doesn't exist
        os.makedirs(artifacts_dir, exist_ok=True)
        
        logging.info(f"Environment setup completed. Artifacts directory: {artifacts_dir}")
    
    
    def _data_ingestion(self):
        """Handle data ingestion process."""
        logging.info("Starting data ingestion")
        try:
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.init_ingestion()
            logging.info(f"Data ingestion completed. Train: {train_data_path}, Test: {test_data_path}")
            return train_data_path, test_data_path
        
        except Exception as e:
            logging.error(f"Data ingestion failed: {str(e)}")
            raise CustomException("Failed during data ingestion stage", sys)
    
    
    def _data_tokenization(self, train_path, test_path):
        """Handle data tokenization process."""
        logging.info("Starting data tokenization")
        try:
            data_tokenizer = DataTokenizer(
                train_path=train_path,
                test_path=test_path,
                model_name=self.base_model_name
            )
            tokenized_dataset = data_tokenizer.init_tokenizer()
            self.tokenizer = data_tokenizer.tokenizer
            logging.info("Data tokenization completed successfully")
            return tokenized_dataset
        
        except Exception as e:
            logging.error(f"Data tokenization failed: {str(e)}")
            raise CustomException("Failed during data tokenization stage", sys)
    
    
    def _model_loading(self):
        """Handle model loading process."""
        logging.info(f"Loading base model: {self.base_model_name}")
        try:
            loader = ModelLoader(self.base_model_name)
            self.model = loader.loader_init()
            logging.info("Model loading completed successfully")
            return self.model
        
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise CustomException("Failed during model loading stage", sys)
    
    
    def _model_training(self, tokenized_dataset):
        """Handle model training process."""
        logging.info("Starting training process")
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model or tokenizer not properly initialized")
            
            trainer = ModelTrainer(
                model=self.model,
                tokenizer=self.tokenizer
            )
            trainer.train(tokenized_dataset)
            logging.info("Training completed successfully")
            
        except Exception as e:
            logging.error(f"Training process failed: {str(e)}")
            raise CustomException("Failed during training stage", sys)
    
    
    def _cleanup_memory(self):
        """Clean up GPU memory and resources."""
        try:
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()  # Move model to CPU first
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force Python garbage collection
            gc.collect()
            logging.info("Memory cleanup completed")
            
        except Exception as e:
            logging.warning(f"Memory cleanup encountered an issue: {str(e)}")
    
    
    def run_pipeline(self):
        """Execute the complete training pipeline."""
        try:
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self._data_ingestion()
            
            # Step 2: Data Tokenization
            tokenized_dataset = self._data_tokenization(train_data_path, test_data_path)
            
            # Step 3: Model Loading
            self._model_loading()
            
            # Step 4: Model Training
            self._model_training(tokenized_dataset)
            
            logging.info("Training pipeline completed successfully!")
            
        except CustomException:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            logging.error(f"Unexpected error in training pipeline: {str(e)}")
            raise CustomException("Unexpected error occurred in training pipeline", sys)
        finally:
            # Always clean up memory
            self._cleanup_memory()


def main():
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()