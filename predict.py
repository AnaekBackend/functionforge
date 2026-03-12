import argparse
import yaml
import json
import os
import torch
from unsloth import FastLanguageModel
from utils import load_tools, build_prompt, parse_function_call

def predict(query, config_path, run_dir=None):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if run_dir is None:
        # Find latest run
        base_dir = config['training']['output_dir']
        runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("run_")]
        if not runs:
            print("No runs found. Please provide a run directory.")
            return
        run_dir = max(runs, key=os.path.getmtime)
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=run_dir,
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
        dtype=None
    )
    FastLanguageModel.for_inference(model)

    # Load tools
    tools = load_tools(config['data']['tools_file'])
    
    # Build prompt
    prompt = build_prompt(query, tokenizer, tools)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response_part = gen_text.split("<start_of_turn>model\n")[-1] if "<start_of_turn>model\n" in gen_text else gen_text
    
    pred = parse_function_call(response_part)
    
    print(json.dumps(pred, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with FunctionGemma model")
    parser.add_argument("query", type=str, help="User query string")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run directory (optional, defaults to latest)")
    args = parser.parse_args()
    predict(args.query, args.config, args.run_dir)
