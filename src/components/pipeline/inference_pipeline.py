import os
import sys
import torch
from transformers import AutoModelForCausalLM
from src.components.config import Config
from src.components.data_tokenizer import DataTokenizer
from peft import PeftModel
from src.logger import logging
from src.exception import CustomException

class InferencePipeline:
    def __init__(self, checkpoint_name=None):
        self.config = Config()
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = self._resolve_checkpoint_dir()
        self.tokenizer = None
        self.model = None


    def _resolve_checkpoint_dir(self):
        if self.checkpoint_name:
            return os.path.join(self.config.output_dir, self.checkpoint_name)
        return self.config.resume_from_checkpoint or self.config.output_dir

    def load_from_checkpoint(self):
        """Load model and tokenizer from a full model checkpoint (Trainer.save_model output)."""
        try:
            logging.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
            self.tokenizer = DataTokenizer().tokenizer_init()
            
            logging.info(f"Loading model from checkpoint: {self.checkpoint_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_dir, torch_dtype=torch.float16, device_map="auto"
            )
            self.model.eval()
            logging.info("Finished loading Model checkpoint and tokenizer.")
        except Exception as e:
            logging.error(f"Error loading from checkpoint: {e}")
            raise CustomException(e, sys)


    def load_from_lora_adapter(self):
        """Load base model and apply LoRA adapter weights (adapter-only export)."""
        try:
            logging.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
            self.tokenizer = DataTokenizer().tokenizer_init()
            
            logging.info(f"Loading base model: {self.config.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, torch_dtype=torch.float16, device_map="auto"
            )
            logging.info(f"Loading LoRA adapter from: {self.checkpoint_dir}")
            self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
            
            self.model.eval()
            logging.info("Model and tokenizer loaded with LoRA adapter.")
        except Exception as e:
            logging.error(f"Error loading from LoRA adapter: {e}")
            raise CustomException(e, sys)

    def generate_response(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.95):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = InferencePipeline()
    # use full model
    # pipeline.load_from_checkpoint()
    
    # use peft adapters
    # pipeline.load_from_lora_adapter()
    print(pipeline.generate_response('Hello!'))