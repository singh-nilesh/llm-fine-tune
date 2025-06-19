"""
This file configures and loads the base model for training, including quantization and LoRA adapter setup.
"""
import os
import sys
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.logger import logging
from src.exception import CustomException
from src.components.config import Config
from src.components.data_tokenizer import DataTokenizer

class ModelLoader:
    def __init__(self):
        self.config = Config()
    
    def loader_init(self, tokenizer=None):
        """Main function to load base model and apply LoRA adapters. Accepts tokenizer for embedding resize."""
        try:
            model = self._load_base_model()
            
            # Resize model embeddings if tokenizer is provided
            if tokenizer is not None:
                model.resize_token_embeddings(len(tokenizer))
                logging.info("Resized model embeddings to match tokenizer vocab size.")
            
            # Check if adapter_config.json exists (proper way to check for valid LoRA adapters)
            adapter_config_path = os.path.join(self.config.train_model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config_path):
                logging.info(f"Loading LoRA adapters from {self.config.train_model_path}")
                model = PeftModel.from_pretrained(model, self.config.train_model_path)
            else:   
                # Create directory if it doesn't exist
                os.makedirs(self.config.train_model_path, exist_ok=True)
                logging.info("Creating new LoRA adapters.")
                lora_config = self._get_lora_config()
                model = get_peft_model(model, lora_config)
                model.save_pretrained(self.config.train_model_path)
                logging.info(f"LoRA adapters saved to {self.config.train_model_path}")
            
            return model
            
        except Exception as e:
            raise CustomException(f"Model loading failed: {e}", sys)
    
    def _load_base_model(self):
        """Loads the base (optionally quantized) model"""
        quant_config = self._get_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_config,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True
        )
        
        # Prepare quantized model if required
        if self.config.use_quantization:
            model = prepare_model_for_kbit_training(model)
        return model
    
    def _get_quantization_config(self):
        """Returns quantization config if enabled"""
        if not self.config.use_quantization:
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype
        )
    
    def _get_lora_config(self):
        """Returns a configured LoraConfig object"""
        try:
            return LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                task_type=self.config.task_type
            )
        except Exception as e:
            raise CustomException(f"Error during LoRA config: {e}", sys)

if __name__ == "__main__":
    tokenizer_inst = DataTokenizer()
    tokenizer = tokenizer_inst.tokenizer_init()
    loader = ModelLoader()
    model = loader.loader_init(tokenizer=tokenizer)