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
        self. model = None
        self.config = Config()
    
    def _setup_environment(self):
        """Set up environment variables and configurations."""
        # Set environment variable for better CUDA OOM handling
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
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
            # Env setup
            self._setup_environment()
            
            # Step 1: Data Ingestion
            ingestor = DataIngestion()
            train_data_path, test_data_path = ingestor.init_ingestion()
            
            # Step 2: Data Tokenization
            tokenizer_obj = DataTokenizer()
            dataset = tokenizer_obj.get_tokenized_data()
            
            # Step 3: Model Loading
            loader = ModelLoader(tokenizer_obj.tokenizer)
            model = loader.loader_init()
            
            # Step 4: Model Training
            trainer = ModelTrainer(model, tokenizer_obj.tokenizer)
            trainer.train(dataset)
            
            logging.info("Training pipeline completed successfully!")
            
        except Exception as e:
            raise CustomException("Unexpected error occurred in training pipeline", sys)
        finally:
            
            # Always clean up memory
            self._cleanup_memory()


def main():
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()