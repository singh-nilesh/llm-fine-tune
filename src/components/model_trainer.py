import os
import sys
from transformers import TrainingArguments
from dataclasses import dataclass
from trl import SFTTrainer

from src.exception import CustomException
from src.logger import logging
from src.components.model_loader import ModelLoaderConfig


@dataclass
class TrainerConfig:
    output_dir: str = os.path.join("artifacts", 'lora_output')

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_config = TrainerConfig()
    
    def train(self, dataset):
        try:
            train_args = TrainingArguments(
                output_dir= self.trainer_config.output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=2,  # batch size per device during training
                gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
                gradient_checkpointing=True,  # use gradient checkpointing to save memory
                optim="adamw_torch_fused",  # use fused adamw optimizer
                logging_steps=10,
                save_strategy="epoch",
                bf16=True,  # use bfloat16 precision
                tf32=True,  # use tf32 precision
                learning_rate=2e-4,  # learning rate, based on QLoRA paper
                max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
                lr_scheduler_type="constant",  # use constant learning rate scheduler
                push_to_hub=False,  # push model to hub
                report_to="tensorboard",  # report metrics to tensorboard
            )
            
            max_seq_length = 3072  # max sequence length for model and packing of the dataset
            
            trainer = SFTTrainer(
                model=self.model,
                args=train_args,
                train_dataset=dataset,
                peft_config=ModelLoaderConfig.lora_config,
                max_seq_length=max_seq_length,
                tokenizer=self.tokenizer,
                packing=True,
                dataset_kwargs={
                    "add_special_tokens": False,  # We template with special tokens
                    "append_concat_token": False,  # No need to add additional separator token
                }
            )
            
            logging.info("Starting training")
            trainer.train()
            
            # Create output directory if it doesn't exist
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            
            # Save the model and tokenizer
            self.model.save_pretrained(self.trainer_config.output_dir)
            self.tokenizer.save_pretrained(self.trainer_config.output_dir)
            
            logging.info(f"Model saved to disk at: {self.trainer_config.output_dir}")
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise CustomException(e, sys)