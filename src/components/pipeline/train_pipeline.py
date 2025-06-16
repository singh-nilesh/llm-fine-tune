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
from src.components.config import Config


class TrainingPipeline:
    """Main training pipeline for model fine-tuning."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.config = Config()
    
    def _setup_environment(self):
        """Set up environment variables and configurations."""
        # Set environment variable for better CUDA OOM handling
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        logging.info("Environment setup completed")
    
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
            
            if self.dataset is not None:
                del self.dataset
                self.dataset = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force Python garbage collection
            gc.collect()
            
            logging.info("Memory cleanup completed")
            
        except Exception as e:
            logging.warning(f"Memory cleanup encountered an issue: {str(e)}")
    
    def main(self):
        """Execute the complete training pipeline."""
        try:
            # Environment setup
            self._setup_environment()
            
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion...")
            ingestor = DataIngestion()
            ingestor.init_ingestion()
            
            # Step 2: Load raw dataset (no tokenization here)
            logging.info("Loading raw dataset for training...")
            tokenizer_inst = DataTokenizer()
            self.tokenizer = tokenizer_inst.tokenizer_init()
            self.dataset = tokenizer_inst.load_data()
            
            # Step 3: Model Loading
            logging.info("Loading model...")
            loader = ModelLoader()
            self.model = loader.loader_init()
            
            # Step 4: Model Training
            logging.info("Starting model training...")
            model_trainer = ModelTrainer()
            model_trainer.train(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.dataset
            )
            
            logging.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(f"Unexpected error occurred in training pipeline: {str(e)}", sys)
        
        finally:
            # Always clean up memory
            self._cleanup_memory()


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.main()