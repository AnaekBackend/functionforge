import argparse
import yaml
import json
import os
import torch
from datetime import datetime
from compat import (
    FastLanguageModel,
    IS_MLX,
    SFTTrainer,
    cuda_precision_kwargs,
    sanitize_lora_kwargs,
    sanitize_training_args_kwargs,
    train_on_responses_only,
)
from transformers import TrainingArguments
from utils import load_tools, prepare_dataset

def train(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['training']['output_dir'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save metadata
    metadata = {
        "config": config,
        "timestamp": timestamp,
        # Add dataset version/hash if available
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
        dtype=None
    )

    # Add LoRA adapters
    lora_cfg = sanitize_lora_kwargs(config["lora"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
    )

    # Load tools
    tools = load_tools(config['data']['tools_file'])

    # Prepare dataset
    train_dataset = prepare_dataset(config['data']['train_file'], tokenizer, tools)

    # Training arguments
    training_cfg = sanitize_training_args_kwargs(config["training"])
    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        **cuda_precision_kwargs(torch),
        logging_steps=1,
        optim=training_cfg["optim"],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        output_dir=run_dir,
        report_to=config['training']['report_to'],
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        dataset_num_proc=2,
        packing=config['training']['packing'],
        args=training_args,
    )

    # Configure to train on responses only.
    # Note: MLX-Tune currently routes this to mlx_lm prompt-masking, which is not
    # supported for our single-column "text" dataset format.
    if not IS_MLX:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

    # Train
    print("Starting training...")
    trainer_stats = trainer.train()
    
    # Save model
    print(f"Saving model to {run_dir}")
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    
    # Save metrics
    metrics = trainer_stats.metrics
    with open(os.path.join(run_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FunctionGemma model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    args = parser.parse_args()
    train(args.config)
