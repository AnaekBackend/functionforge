import json
import re
from pathlib import Path
from datasets import Dataset

# System message for prompt building
SYSTEM_MSG = (
    "You are an AttendanceBot function router. "
    "You MUST choose exactly one tool based on the user's message. "
    "If the request is ambiguous or invalid, use ask_clarify. "
    "Extract all relevant parameters from the user message."
)

def load_tools(tools_file_path):
    """Load tools from JSON file."""
    with open(tools_file_path, 'r') as f:
        tools = json.load(f)
    return tools

def build_prompt(user_text: str, tokenizer, tools) -> str:
    """Build prompt using apply_chat_template with tools."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,   # Leaves model turn open
        tokenize=False,
    )
    return rendered

def gold_to_function_call(gold):
    """
    Convert JSON output to FunctionGemma format.
    
    Input: {"name": "apply_leave", "arguments": {"leave_type": "vacation", "start_date": "tomorrow"}}
    Output: <start_function_call>call:apply_leave{leave_type:<escape>vacation<escape>,start_date:<escape>tomorrow<escape>}<end_function_call>
    """
    name = gold["name"]
    args = gold["arguments"]
    
    # Build key-value pairs
    kv_parts = []
    for k, v in args.items():
        if isinstance(v, int):
            kv_parts.append(f"{k}:{v}")
        else:
            kv_parts.append(f"{k}:<escape>{v}<escape>")
    
    kv = ",".join(kv_parts)
    return f"<start_function_call>call:{name}{{{kv}}}<end_function_call>"

def row_to_text(row, tokenizer, tools):
    """Convert data row to formatted text for training."""
    user_text = row["input"]
    tool_call = gold_to_function_call(row["output"])
    
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_text},
        # Assistant content is the exact function call
        {"role": "assistant", "content": tool_call},
    ]
    
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,  # Assistant content included
        tokenize=False,
    )
    return rendered

CALL_RE = re.compile(r"call:([a-zA-Z0-9_]+)\{(.*?)\}", re.DOTALL)

def unescape(s: str) -> str:
    return s.replace("<escape>", "")

def parse_function_call(text: str):
    """
    Parse FunctionGemma output to JSON.
    
    Returns: {"name": str, "arguments": dict} or None
    """
    m = CALL_RE.search(text)
    if not m:
        return None
    
    name = m.group(1).strip()
    args_blob = unescape(m.group(2)).strip()
    
    # Parse key:value pairs
    args = {}
    parts = [p.strip() for p in args_blob.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip().strip('"').strip("'")
        v = v.strip().strip('"').strip("'")
        
        # Convert to int if numeric
        if v.isdigit():
            args[k] = int(v)
        else:
            args[k] = v
    
    return {"name": name, "arguments": args}

def load_dataset(file_path):
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows

def prepare_dataset(file_path, tokenizer, tools):
    rows = load_dataset(file_path)
    texts = [row_to_text(r, tokenizer, tools) for r in rows]
    return Dataset.from_dict({"text": texts})
