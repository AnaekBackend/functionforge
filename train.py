import argparse
import yaml
import json
import os
import torch
from datetime import datetime
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only
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
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing=config['lora']['use_gradient_checkpointing'],
        random_state=config['lora']['random_state'],
    )

    # Load tools
    tools = load_tools(config['data']['tools_file'])

    # Prepare dataset
    train_dataset = prepare_dataset(config['data']['train_file'], tokenizer, tools)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim=config['training']['optim'],
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

    # Configure to train on responses only
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
