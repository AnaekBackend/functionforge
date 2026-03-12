# macOS (Apple Silicon) / MLX-Tune Support Notes

This repo was originally built around **Unsloth (CUDA)** and **TRL** and therefore didn’t run on Macs without an NVIDIA GPU.

We added support for **Apple Silicon** using **[MLX-Tune](https://github.com/ARahim3/mlx-tune)** (an Unsloth-compatible API on top of Apple’s MLX).

This document lists *exactly* what we changed and why.

## Summary of changes

- **Backend selection**: Added a small compatibility layer (`compat.py`) that selects **Unsloth+TRL** on non-macOS and **MLX-Tune** on macOS.
- **Eval/predict**: MLX tokenization/generation APIs differ from HuggingFace/torch, so `eval.py` and `predict.py` now use an MLX-specific generation path.
- **Baseline eval**: `eval.py` can evaluate the *base model* before any fine-tuning runs exist (`results/base_<timestamp>/`).
- **Training**:
  - MLX prompt masking (response-only training) is **not supported** for a single `"text"` dataset column.
  - We changed the MLX dataset path to use **`prompt` + `completion`** columns so response-only training works.
  - MLX-Tune’s `trainer.train()` returns a **dict**, not a Transformers `TrainOutput`, so metrics saving was updated to support both.
- **Dependencies**:
  - Added `mlx-tune` on macOS; gated `unsloth` + `trl` off on macOS.
  - Added `tabulate` (needed for `pandas.DataFrame.to_markdown()` in the eval report).

## 1) Compatibility layer (`compat.py`)

### What it does

`compat.py`:
- detects macOS via `sys.platform == "darwin"` (exported as `IS_MLX`)
- imports the correct backend symbols:
  - macOS: `from mlx_tune import FastLanguageModel, SFTTrainer, train_on_responses_only`
  - other OS: `from unsloth import FastLanguageModel`, `from trl import SFTTrainer`, `from unsloth.chat_templates import train_on_responses_only`

### Why it exists

MLX-Tune’s promise is “same API, different import”. In practice, we still needed a couple of callsite-level differences, but the shim keeps those differences localized and keeps the rest of the code readable.

### Normalization helpers

`compat.py` also sanitizes a few config knobs that are CUDA/Unsloth-specific:

- **Gradient checkpointing**:
  - config value `"unsloth"` is an Unsloth sentinel
  - on MLX we coerce it to `True`
- **Optimizer**:
  - `"adamw_8bit"` can be CUDA-only
  - on MLX we swap it to `"adamw_torch"`
- **Precision flags**:
  - CUDA uses `fp16/bf16` flags
  - on MLX we skip those entirely

## 2) Eval + predict: tokenizer / generate differences

### The issue

On CUDA we do:

- tokenize with HuggingFace tokenizer: `tokenizer(prompt, return_tensors="pt")`
- send tensors to device: `.to(model.device)`
- generate with `max_new_tokens`, `temperature`, etc.
- decode tokens

On MLX:
- the returned tokenizer is a `TokenizerWrapper` and **is not callable**
- `MLXModelWrapper.generate()` is a wrapper around `mlx_lm.generate()` and expects:
  - `prompt=...`
  - `max_tokens=...` (not `max_new_tokens`)
  - and returns the **completion text directly** (not token IDs)

### The fix

In `eval.py` and `predict.py`, we branch on `IS_MLX`:

- **MLX path**: `response_part = model.generate(prompt=prompt, max_tokens=128)`
- **CUDA path**: keep the original HF/torch tokenization + `max_new_tokens` generation + decode

## 3) Baseline eval before finetune

### Goal

Allow this workflow:

1. **Eval base model**
2. Fine-tune
3. **Eval fine-tuned model**

### Change

If `--run_dir` is not provided and there’s no `results/run_*` directory yet, `eval.py` now:
- evaluates `config.model.name` directly
- writes reports into `results/base_<timestamp>/`

## 4) Training: enabling response-only masking on MLX

### Root cause

MLX-LM datasets support prompt masking (`mask_prompt=True`) only for:

- **Completions dataset**: rows contain **both** `prompt` and `completion`
- **Chat dataset**: rows contain `messages`

If you supply only `{"text": ...}` then MLX-LM treats it as a **TextDataset** and raises:

> `ValueError: Prompt masking not supported for text dataset.`

### Fix: emit `prompt` + `completion` on MLX

We updated `utils.prepare_dataset()` to optionally return:

```json
{"prompt": "...", "completion": "..."}
```

Implementation detail:
- `prompt` is produced using `build_prompt(...)` (system+user with generation prompt open)
- `completion` is extracted by computing `full = row_to_text(...)` and taking `full[len(prompt):]`
  - with a safe fallback if template mismatches

In `train.py`, MLX now calls:

- `prepare_dataset(..., as_prompt_completion=True)`
- `dataset_text_field=None` (MLX-Tune/MLX-LM will auto-detect completions)

This makes `train_on_responses_only()` work on MLX again without falling back to legacy subprocess training.

## 5) Training: metrics return type differences

### The issue

- TRL/Transformers typically returns a `TrainOutput` with `.metrics`
- MLX-Tune returns a **dict** (e.g. `{"status": "...", "adapter_path": "..."}`)

### Fix

`train.py` now writes `metrics.json` with:
- the dict directly if `trainer_stats` is a dict
- otherwise `trainer_stats.metrics`

## 6) Dependency changes

### `pyproject.toml` / `requirements.txt`

- Added `mlx-tune` with `sys_platform == "darwin"`
- Gated `unsloth`, `unsloth-zoo`, and `trl` behind `sys_platform != "darwin"`
- Added `tabulate` (required by `pandas.DataFrame.to_markdown()` used for `eval_report.md`)

## 7) Model name changes for MLX

The CUDA default model was:
- `unsloth/functiongemma-270m-it`

On macOS, you typically need an MLX-converted model, e.g.:
- `mlx-community/functiongemma-270m-it-4bit`

## Notes / gotchas

- **HF token**: HuggingFace downloads work unauthenticated but are slower / rate-limited. Setting `HF_TOKEN` improves UX.
- **Parity**: MLX-Tune is highly compatible, but not perfectly identical:
  - tokenizer callability and generation kwargs differ
  - trainer return types differ
  - prompt-masking constraints differ based on dataset format

