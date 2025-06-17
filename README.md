# llm-fine-tune

## Project Overview
This project provides a modular pipeline for fine-tuning Large Language Models (LLMs) using HuggingFace, LoRA, and quantization techniques. It is designed for robust, reproducible, and scalable training, with all artifacts optionally stored on Google Drive for persistence.

---

## Pipeline Progression

1. **Data Ingestion**
   - `src/components/data_ingestion.py`: Connects to Supabase, fetches raw records, formats them into instruction-following Q&A pairs, and saves them as `.jsonl` files in the artifacts directory.

2. **Data Tokenization**
   - `src/components/data_tokenizer.py`: Loads the tokenizer (with special tokens if needed) and loads the raw dataset from `.jsonl` files. No pre-tokenization is performed; tokenization is handled during training.

3. **Model Loading**
   - `src/components/model_loader.py`: Loads the base model (optionally with quantization), applies LoRA adapters, and resizes the model's embedding matrix to match the tokenizer vocabulary. Ensures compatibility between tokenizer and model.

4. **Model Training**
   - `src/components/model_trainer.py`: Sets up the SFTTrainer with the model, tokenizer, and dataset. Handles experiment tracking (WandB), training arguments, and saving the final model.

5. **Pipeline Orchestration**
   - `src/components/pipeline/train_pipeline.py`: Orchestrates the full pipeline: environment setup, data ingestion, tokenizer/model loading, training, and memory cleanup. Handles errors and ensures reproducibility.

---
### Configuration & Utilities
- **`src/components/config.py`** — Central configuration for paths, hyperparameters, and Google Drive integration for artifacts.
- **`src/logger.py`** — Logging utility for consistent log formatting.
- **`src/exception.py`** — Custom exception class for robust error handling.
- **`src/utils.py`** — (If present) Shared utility functions.

### Artifacts & Outputs
- **`src/components/artifacts/`** — Stores all outputs: processed data, tokenizers, model checkpoints, LoRA adapters, and logs. Can be redirected to Google Drive for persistence.

### Project Root
- **`README.md`** — Project documentation and usage instructions.
- **`requirements.txt`** — Python dependencies.
- **`setup.py`** — (Optional) For pip-installable package.

---

## How to Run the Pipeline
1. Set up your environment variables (Supabase, WandB, etc.).
2. Run the pipeline:
   ```bash
   python src/components/pipeline/train_pipeline.py
   ```
3. All artifacts and outputs will be saved in the `artifacts` directory (local or on Google Drive).

---

## Notes
- The pipeline is modular: you can run or test each component independently.
- All paths and hyperparameters are managed via `src/components/config.py`.
- For inference or evaluation, add scripts such as `inference.py` or `model_evaluator.py` as needed.
