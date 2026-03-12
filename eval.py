import argparse
import yaml
import json
import os
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from utils import load_tools, build_prompt, parse_function_call, load_dataset
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import jsonschema
import pandas as pd

def compare(pred, gold):
    """Compare predicted vs gold function call."""
    res = {
        "has_call": 0,
        "tool_ok": 0,
        "args_exact": 0,
        "field_acc": {},
    }
    
    if pred is None:
        return res
    
    res["has_call"] = 1
    res["tool_ok"] = int(pred.get("name") == gold.get("name"))
    
    pred_args = pred.get("arguments", {}) or {}
    gold_args = gold.get("arguments", {}) or {}
    
    res["args_exact"] = int(res["tool_ok"] == 1 and pred_args == gold_args)
    
    # Per-field accuracy
    for k, gv in gold_args.items():
        res["field_acc"][k] = int(pred_args.get(k, None) == gv)
    
    return res

def evaluate(config_path, run_dir=None):
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
    
    print(f"Evaluating run: {run_dir}")

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
    
    # Load eval dataset
    eval_rows = load_dataset(config['data']['eval_file'])

    # Build schema map for validation
    schema_map = {t['function']['name']: t['function'] for t in tools}

    stats = defaultdict(float)
    failures = []
    y_true = []
    y_pred = []
    
    print("Generating predictions...")
    for row in tqdm(eval_rows):
        user_text = row["input"]
        gold = row["output"]
        
        prompt = build_prompt(user_text, tokenizer, tools)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract the assistant response part
        response_part = gen_text.split("<start_of_turn>model\n")[-1] if "<start_of_turn>model\n" in gen_text else gen_text
        
        pred = parse_function_call(response_part)
        
        r = compare(pred, gold)
        
        stats["count"] += 1
        stats["has_call"] += r["has_call"]
        stats["tool_ok"] += r["tool_ok"]
        stats["args_exact"] += r["args_exact"]
        
        # Schema Validation
        valid_schema = False
        if pred and pred['name'] in schema_map:
            try:
                # Construct a dummy instance to validate against the schema parameters
                # Note: FunctionGemma format flattens arguments, but schema expects an object
                jsonschema.validate(instance=pred['arguments'], schema=schema_map[pred['name']]['parameters'])
                valid_schema = True
            except jsonschema.exceptions.ValidationError:
                valid_schema = False
        stats["valid_schema"] += int(valid_schema)

        y_true.append(gold.get("name", "None"))
        y_pred.append(pred.get("name", "None") if pred else "None")

        if r["args_exact"] == 0:
            failures.append({
                "input": user_text,
                "gold": gold,
                "pred": pred,
                "raw": response_part
            })

    # Compute Metrics
    accuracy = stats["args_exact"] / stats["count"] if stats["count"] > 0 else 0
    invalid_rate = 1 - (stats["has_call"] / stats["count"]) if stats["count"] > 0 else 0
    schema_validity_rate = stats["valid_schema"] / stats["count"] if stats["count"] > 0 else 0
    
    report = {
        "overall_accuracy": accuracy,
        "invalid_function_call_rate": invalid_rate,
        "json_schema_validity_rate": schema_validity_rate,
        "total_samples": stats["count"],
        "tool_selection_accuracy": stats["tool_ok"] / stats["count"] if stats["count"] > 0 else 0
    }

    # Classification Report
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    report["per_intent_metrics"] = clf_report

    # Save JSON report
    with open(os.path.join(run_dir, "eval_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
        
    # Save Top 20 Failures
    with open(os.path.join(run_dir, "failures.json"), 'w') as f:
        json.dump(failures[:20], f, indent=4)

    # Generate Markdown Report
    md_report = f"# Evaluation Report\n\n"
    md_report += f"**Run Directory:** `{run_dir}`\n\n"
    md_report += f"## Overall Metrics\n"
    md_report += f"- **Overall Accuracy (Exact Match):** {accuracy:.2%}\n"
    md_report += f"- **Tool Selection Accuracy:** {report['tool_selection_accuracy']:.2%}\n"
    md_report += f"- **Invalid Function Call Rate:** {invalid_rate:.2%}\n"
    md_report += f"- **JSON Schema Validity Rate:** {schema_validity_rate:.2%}\n\n"
    
    md_report += f"## Per-Intent Metrics\n"
    df_clf = pd.DataFrame(clf_report).transpose()
    md_report += df_clf.to_markdown() + "\n\n"
    
    md_report += f"## Top 20 Failures\n"
    for i, fail in enumerate(failures[:20]):
        md_report += f"### Failure {i+1}\n"
        md_report += f"- **Input:** {fail['input']}\n"
        md_report += f"- **Gold:** `{json.dumps(fail['gold'])}`\n"
        md_report += f"- **Pred:** `{json.dumps(fail['pred'])}`\n"
        md_report += f"- **Raw:** `{fail['raw']}`\n\n"

    with open(os.path.join(run_dir, "eval_report.md"), 'w') as f:
        f.write(md_report)

    print(f"Evaluation complete. Reports saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run directory (optional, defaults to latest)")
    args = parser.parse_args()
    evaluate(args.config, args.run_dir)
