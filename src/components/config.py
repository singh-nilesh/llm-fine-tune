"""
Contains configurations for training pipeline
"""
from dataclasses import dataclass, field
from peft import TaskType
from typing import List
import os
import torch

@dataclass
class Config:
    """Configuration class for fine-tuning parameters"""
    
    # Set artifacts directory path
    artifacts_dir: str = os.path.join(os.path.dirname(__file__), "artifacts")
    
    # Model parameters
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512
    
    # LoRA parameters
    use_quantization:bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias:str = "none"
    task_type:TaskType = TaskType.CAUSAL_LM 
    
    # Training parameters
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # System parameters
    seed: int = 42
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    
    # Trainer config
    use_wandb:bool = True
    
    # 4-bit BNB Config
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant:bool = True
    
    # WandB config
    wandb_project: str = "tinyllama-finetune"
    wandb_run_name: str = "tinyllama-lora-4bit"
    wandb_log_model: bool = True
    
    # Test size
    test_size: float = 0.15
    
    def __post_init__(self):
        """Set up derived paths after initialization"""
        # Create artifacts directory if it doesn't exist
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Log the artifacts directory path to confirm it's correct
        from src.logger import logging
        logging.info(f"Using artifacts directory: {os.path.abspath(self.artifacts_dir)}")
        
        self.output_dir = os.path.join(self.artifacts_dir, "lora_output")
        self.lora_adapter_path = os.path.join(self.artifacts_dir, "lora_adapter")
        self.tokenizer_path = os.path.join(self.artifacts_dir, "tokenizer")
        self.tokenize_data_path = os.path.join(self.artifacts_dir, "tokenized_data")
        self.train_data_path = os.path.join(self.artifacts_dir, "train.jsonl")
        self.eval_data_path = os.path.join(self.artifacts_dir, "eval.jsonl")
        
        # Create all required directories
        for directory in [
            self.output_dir,
            self.lora_adapter_path,
            self.tokenizer_path,
            self.tokenize_data_path
        ]:
            os.makedirs(directory, exist_ok=True)
    
    def get_config_info(self):
        """Convert config to dictionary for logging"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }