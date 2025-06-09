"""
    This file configures and loads the base model for training
"""
import os
import torch
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelLoaderConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.LoaderConfig = ModelLoaderConfig()
    
    def loader_init(self):
        # load base model
        qa_model = self.load_base_model()
        
        # Add LoRA adapter
        model = get_peft_model(qa_model, self.LoaderConfig.lora_config)
        
        logging.info(f"trainable parameters \n {model.print_trainable_parameters()}")
        return model


    def load_base_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.LoaderConfig.bnb_config,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(base_model)
        logging.info("quantized base model")
        return model

if __name__ == "__main__":
    pass