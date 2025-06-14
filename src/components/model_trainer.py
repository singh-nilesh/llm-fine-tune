import os
import sys
import torch
from transformers import TrainingArguments
from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
import wandb
from dotenv import load_dotenv  # Add this import
from src.exception import CustomException
from src.logger import logging
from src.components.model_loader import ModelLoaderConfig

# Load environment variables from .env file
load_dotenv()

@dataclass
class TrainerConfig:
    output_dir: str = os.path.join("artifacts", 'lora_output')
    use_wandb: bool = True
    project_name: str = "model_training"
    run_name: str = "lora_training_run"

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_config = TrainerConfig()
    
    def initialize_wandb(self):
        """Initialize wandb for experiment tracking"""
        if not self.trainer_config.use_wandb:
            logging.info("Wandb tracking disabled by configuration")
            return False
        
        try:
            # Use API key from environment variables
            wandb_api_key = os.environ.get("WANDB_KEY")
            
            if not wandb_api_key:
                logging.warning("WANDB_KEY not found in environment variables")
                return False
                
            wandb.login(key=wandb_api_key)
            
            wandb.init(
                project=self.trainer_config.project_name,
                name=self.trainer_config.run_name,
                config={
                    "model_name": self.model.config._name_or_path,
                    "epochs": 3,
                    "batch_size": 3,
                    "learning_rate": 2e-4,
                }
            )
            logging.info("Wandb initialized successfully")
            return True
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {str(e)}. Continuing without wandb tracking.")
            return False
    
    def train(self, train_data, test_data):
        try:
            # Check if TF32 is supported
            use_tf32 = False
            report_to = ["tensorboard"]
            
            try:
                if torch.cuda.is_available():
                    device_capability = torch.cuda.get_device_capability()
                    cuda_version = torch.version.cuda
                    
                    # TF32 requires Ampere or newer (compute capability >= 8.0) and CUDA >= 11
                    if device_capability[0] >= 8 and cuda_version is not None and cuda_version.split('.')[0] >= '11':
                        use_tf32 = True
                        logging.info("TF32 precision is supported and will be enabled")
                    else:
                        logging.info(f"TF32 not supported: GPU capability {device_capability}, CUDA version {cuda_version}")
            except Exception as e:
                logging.warning(f"Error checking TF32 compatibility: {str(e)}. TF32 will be disabled.")
            
            # Initialize wandb if enabled
            if self.initialize_wandb():
                report_to.append("wandb")
 
            args = TrainingArguments(
                output_dir=self.trainer_config.output_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=8,
                num_train_epochs=3,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                logging_steps=10,
                save_strategy="steps",
                save_steps=100,
                eval_strategy="steps",
                eval_steps=100,
                bf16=True,
                gradient_checkpointing=True,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=report_to,
                seed=42,
            )

            trainer = SFTTrainer(
                model=self.model,
                args=args,
                train_dataset=train_data,
                peft_config=ModelLoaderConfig.lora_config,
                tokenizer=self.tokenizer,
            )
            
            logging.info("Starting training")
            trainer.train()
            
            # Create output directory if it doesn't exist
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            
            # Save the model and tokenizer
            self.model.save_pretrained(self.trainer_config.output_dir)
            self.tokenizer.save_pretrained(self.trainer_config.output_dir)
            
            logging.info(f"Model saved to disk at: {self.trainer_config.output_dir}")
            
            # Finish wandb tracking if initialized
            if wandb.run is not None:
                wandb.finish()
    
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            # Make sure to finish the wandb run in case of error
            if wandb.run is not None:
                wandb.finish()
            raise CustomException(e, sys)