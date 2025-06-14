"""
Contains configurations for training pipeline
"""
from dataclasses import dataclass, field
from typing import List
import os
import torch

@dataclass
class Config:
    """Configuration class for fine-tuning parameters"""
    
    # Set artifacts directory at the same level as this file
    artifacts_dir: str = "artifacts"
    
    # Model parameters
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
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
    use_wandb: bool = True
    project_name: str = "model_training"
    run_name: str = "lora_training_run"
    
    # 4-bit BNB Config
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"
    
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
        
        self.output_dir = os.path.join(self.artifacts_dir, "lora_output")
        self.tokenizer_path = os.path.join(self.artifacts_dir, "tokenizer")
        self.tokenize_data_path = os.path.join(self.artifacts_dir, "tokenized_data")
        self.train_data_path = os.path.join(self.artifacts_dir, "train.jsonl")
        self.eval_data_path = os.path.join(self.artifacts_dir, "eval.jsonl")
    
    def get_config_info(self):
        """Convert config to dictionary for logging"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }