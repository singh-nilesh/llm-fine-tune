import os
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class TrainerConfig:
    output_dir = os.path.join("artifacts", 'lora_output')

class Trainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.TrainerConfig = TrainerConfig()
    
    
    def train(self, dataset):
        
        training_args = TrainingArguments(
        output_dir=self.TrainerConfig.output_dir,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy='no',
        bf16=True
        )
        
        trainer = Trainer(
            model= self.model,
            args = training_args,
            train_dataset = dataset,
            tokenizer=self.tokenizer
        )
        
        logging.info("Starting training")
        trainer.train()
        
        self.model.save_pretrained(self.TrainerConfig.output_dir)
        self.tokenizer.save_pretrained(self.TrainerConfig.output_dir)
        logging.info("Model save to disk")
