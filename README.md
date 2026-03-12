# FunctionForge

Fine-tune [FunctionGemma-270M](https://huggingface.co/unsloth/functiongemma-270m-it) to route natural-language attendance messages to structured function calls. Uses [Unsloth](https://github.com/unslothai/unsloth) for 4-bit QLoRA training.

## macOS (Apple Silicon) support

This repo now supports running the same workflow on Apple Silicon via [MLX-Tune](https://github.com/ARahim3/mlx-tune) (an Unsloth-compatible API on top of Apple’s MLX).

- On **macOS**, install pulls in `mlx-tune` and skips `unsloth`/`trl`.
- You’ll also want to set `model.name` in `configs/default.yaml` to an MLX-compatible base model (typically `mlx-community/*` on HuggingFace). Keeping `unsloth/functiongemma-270m-it` will work on CUDA but usually won’t on MLX.

## Task

Given a short user message like `"take a sick day tomorrow"`, the model must output a structured function call:

```json
{"name": "apply_leave", "arguments": {"leave_type": "sick", "start_date": "tomorrow"}}
```

Nine tools are supported: `apply_leave`, `clock_in`, `clock_out`, `get_summary`, `get_timesheet`, `change_manager`, `take_break`, `get_help`, `ask_clarify`.

## Repo Structure

```text
.
├── train.py              # Fine-tuning script (QLoRA via Unsloth + TRL)
├── eval.py               # Evaluation script (exact match, tool accuracy, schema validity)
├── predict.py            # Single-example inference script
├── utils.py              # Prompt building, dataset loading, output parsing
├── configs/
│   └── default.yaml      # Model, LoRA, training, and data configuration
├── data/
│   ├── attendancebot_tools.json   # Tool schemas (OpenAI function-calling format)
│   └── processed/
│       ├── train_function_calling.jsonl
│       └── eval_function_calling.jsonl
├── sample_dataset/
│   ├── train.jsonl       # 30 sample training examples
│   ├── eval.jsonl        # 15 sample eval examples
│   └── tools.json        # Tool schemas
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Quickstart

**Install dependencies (requires [uv](https://docs.astral.sh/uv/)):**

```bash
uv sync
```

**Train:**

```bash
uv run python train.py --config configs/default.yaml
```

**Evaluate** (auto-selects the latest run in `results/`):

```bash
uv run python eval.py --config configs/default.yaml
```

Or point at a specific run directory:

```bash
uv run python eval.py --config configs/default.yaml --run_dir results/run_20240101_120000
```

## Configuration

Edit `configs/default.yaml` to change the model, LoRA rank, training steps, or data paths.

Key fields:

| Field | Default | Description |
|---|---|---|
| `model.name` | `unsloth/functiongemma-270m-it` | Base model |
| `model.load_in_4bit` | `true` | 4-bit quantization |
| `lora.r` | `16` | LoRA rank |
| `training.max_steps` | `150` | Training steps |
| `training.learning_rate` | `2e-4` | Peak LR |
| `data.train_file` | `data/processed/train_function_calling.jsonl` | Training data |
| `data.eval_file` | `data/processed/eval_function_calling.jsonl` | Eval data |

## Dataset Format

Each `.jsonl` line is a JSON object with `input` (user message) and `output` (target function call):

```jsonl
{"input": "clock in", "output": {"name": "clock_in", "arguments": {}}}
{"input": "sick today", "output": {"name": "apply_leave", "arguments": {"leave_type": "sick", "start_date": "today"}}}
{"input": "who's off tomorrow", "output": {"name": "get_summary", "arguments": {"scope": "team", "date": "tomorrow"}}}
```

See `sample_dataset/` for examples.

## Eval Metrics

| Metric | Description |
|---|---|
| Overall Accuracy | Exact match on tool name + all arguments |
| Tool Selection Accuracy | Correct tool name regardless of arguments |
| Invalid Function Call Rate | Fraction of outputs that couldn't be parsed |
| JSON Schema Validity Rate | Fraction of predictions that pass the tool schema |

Per-intent precision/recall/F1 is also reported via `sklearn.metrics.classification_report`.

Results are written to the run directory as `eval_report.json`, `eval_report.md`, and `failures.json`.

## Requirements

- Python ≥ 3.10
- CUDA GPU (tested on A100/T4; 4-bit quantization reduces VRAM to ~4 GB for the 270M model)
