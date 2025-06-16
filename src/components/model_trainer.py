import sys
import os
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import wandb
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging
from src.components.config import Config

# Load environment variables from .env file
load_dotenv()

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.report_to = "tensorboard"
        
        self.wandb_key = os.environ.get("WANDB_KEY")
        if not self.wandb_key:
            logging.warning("wandb_key not found in environment variables")
    
    
    def train(self, model, tokenizer, dataset) -> SFTTrainer:
        """Complete training pipeline with wandb integration"""
        try:
            # Initialize wandb
            self._initialize_wandb()
            
            # Initialize trainer
            logging.info("Initializing SFT Trainer")
            trainer = SFTTrainer(
                model=model,
                args=self._get_training_args(),
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                tokenizer=tokenizer,
                data_collator=self._data_collator(tokenizer),
                dataset_text_field="text",
                max_seq_length=self.config.max_length,
                packing=False,
            )
            
            # Start training
            logging.info("Starting model training")
            trainer.train()
            
            # Save the final model
            trainer.save_model()
            logging.info(f"Model saved to {self.config.output_dir}")
            
            
            logging.info("Training completed successfully")
            return trainer
            
        except Exception as e:
            raise CustomException(e, sys)
        finally:
            # Clean up wandb run even if training fails
            if self.report_to == "wandb":
                wandb.finish()
            
        
        
    def _data_collator(self, tokenizer) -> DataCollatorForLanguageModeling:
        """Data collator for unifying input string length"""
        try:
            logging.info("Data collator compiled")
            return DataCollatorForLanguageModeling(
                tokenizer= tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        except Exception as e:
            logging.error(f"Error creating data collator: {str(e)}")
            raise CustomException(e, sys)
    
    
    def _initialize_wandb(self):
        """Initialize wandb for experiment tracking"""
        if not self.config.use_wandb:
            logging.info("Wandb tracking disabled by configuration")
            return False
        
        try:
            # Use API key from environment variables
            wandb.login(key=self.wandb_key)
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "model_name": self.config.model_name,
                    "epochs": self.config.num_train_epochs,
                    "batch_size": self.config.per_device_train_batch_size,
                    "learning_rate": self.config.learning_rate,
                }
            )
            logging.info("Wandb initialized successfully")
            self.report_to = "wandb"
            return True
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {str(e)}. Continuing tensorboard.")
            return False
        
    
    def _get_training_args(self) -> TrainingArguments:
        """This function returns Training arguments"""
        try:
            logging.info("Creating training arguments")
            return TrainingArguments(
                output_dir= self.config.output_dir,
                per_device_train_batch_size= self.config.per_device_train_batch_size,
                per_device_eval_batch_size= self.config.per_device_eval_batch_size,
                gradient_accumulation_steps= self.config.gradient_accumulation_steps,
                num_train_epochs= self.config.num_train_epochs,
                learning_rate= self.config.learning_rate,
                lr_scheduler_type= self.config.lr_scheduler_type,
                warmup_ratio= self.config.warmup_ratio,
                logging_steps= self.config.logging_steps,
                save_strategy= self.config.save_strategy,
                save_steps= self.config.save_steps,
                evaluation_strategy= self.config.eval_strategy,  # Fixed parameter name
                eval_steps= self.config.eval_steps,
                bf16= self.config.bf16,
                gradient_checkpointing= self.config.gradient_checkpointing,
                dataloader_pin_memory= self.config.dataloader_pin_memory,
                remove_unused_columns= self.config.remove_unused_columns,
                report_to= self.report_to,
                seed= self.config.seed,
            )
        except Exception as e:
            logging.error(f"Error creating training arguments: {str(e)}")
            raise CustomException(e, sys)
