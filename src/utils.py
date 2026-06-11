"""Shared utilities for the depth study benchmark."""
from __future__ import annotations

import json
import re
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=SyntaxWarning)

MODEL_ALIASES = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-30b-moe": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma4-4b": "google/gemma-4-E4B-it",
    "gemma4-26b-moe": "google/gemma-4-26B-A4B-it",
}

AGENT_SYSTEM = (
    "You are a helpful assistant with access to tools. "
    "Use them when appropriate to help the user."
)


def resolve_model_path(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def get_model_family(model_path: str) -> str:
    """Detect model family from path."""
    p = model_path.lower()
    if "gemma" in p:
        return "gemma"
    if "llama" in p:
        return "llama"
    if "qwen3.5" in p or "qwen3_5" in p:
        return "qwen3.5"
    if "qwen" in p:
        return "qwen"
    return "unknown"


def is_native_tool_format(model_path: str) -> bool:
    """Llama 3.1+ and Gemma 4 use native function calling (tool role)."""
    family = get_model_family(model_path)
    return family in ("llama", "gemma")


def _parse_gemma_args(raw: str) -> dict:
    """Parse Gemma's key-value args format with <|"|> string delimiters."""
    args = {}
    raw = raw.replace('<|"|>', '"')
    try:
        d = json.loads("{" + raw + "}")
        return d
    except Exception:
        pass
    for m in re.finditer(r'(\w+)\s*:\s*"([^"]*)"', raw):
        args[m.group(1)] = m.group(2)
    for m in re.finditer(r'(\w+)\s*:\s*(\d+(?:\.\d+)?)', raw):
        if m.group(1) not in args:
            try:
                args[m.group(1)] = float(m.group(2)) if '.' in m.group(2) else int(m.group(2))
            except ValueError:
                pass
    for m in re.finditer(r'(\w+)\s*:\s*(true|false)', raw, re.IGNORECASE):
        if m.group(1) not in args:
            args[m.group(1)] = m.group(2).lower() == "true"
    return args


def parse_all_gemma_tool_calls(text: str, processor=None) -> list:
    """Extract all Gemma 4 tool calls from text.

    If processor is provided, uses processor.parse_response() for robust parsing.
    Falls back to regex extraction otherwise.
    """
    if processor is not None:
        try:
            parsed = processor.parse_response(text)
            tool_calls = parsed.get("tool_calls", [])
            results = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    results.append({"name": name, "arguments": args})
            if results:
                return results
        except Exception:
            pass

    results = []
    for match in re.finditer(r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>', text, re.DOTALL):
        name = match.group(1)
        args = _parse_gemma_args(match.group(2).strip())
        results.append({"name": name, "arguments": args})
    if not results:
        for match in re.finditer(r'call:(\w+)\{(.*?)\}', text, re.DOTALL):
            name = match.group(1)
            args = _parse_gemma_args(match.group(2).strip())
            results.append({"name": name, "arguments": args})
    return results


def parse_qwen35_tool_calls(text: str) -> list:
    """Parse Qwen3.5 XML-style tool calls: <function=name><parameter=k>v</parameter></function>"""
    results = []
    # Find all <tool_call>...</tool_call> blocks first
    blocks = text.split("<tool_call>")
    for block in blocks[1:]:
        body = block.split("</tool_call>", 1)[0].strip()
        # Parse <function=name>...</function>
        fn_match = re.search(r'<function=(\w+)>(.*?)</function>', body, re.DOTALL)
        if not fn_match:
            continue
        name = fn_match.group(1)
        params_str = fn_match.group(2)
        args = {}
        for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_str, re.DOTALL):
            key = pm.group(1)
            val = pm.group(2).strip()
            # Try to parse as JSON value
            try:
                args[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                args[key] = val
        results.append({"name": name, "arguments": args})

    if not results:
        # Fallback: look for <function=...> without <tool_call> wrapper
        for fn_match in re.finditer(r'<function=(\w+)>(.*?)</function>', text, re.DOTALL):
            name = fn_match.group(1)
            params_str = fn_match.group(2)
            args = {}
            for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_str, re.DOTALL):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    args[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    args[key] = val
            results.append({"name": name, "arguments": args})

    return results


def _parse_one_tool_block(body: str) -> Optional[dict]:
    """Parse a single tool call JSON body into a dict with 'name' and 'arguments'."""
    body = body.replace("```json", "").replace("```", "").strip()
    try:
        d = json.loads(body)
        if isinstance(d, dict) and "name" in d:
            d.setdefault("arguments", d.pop("parameters", {}))
            if not isinstance(d.get("arguments"), dict):
                d["arguments"] = {}
            return d
    except Exception:
        pass
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', body)
    if name_match:
        args = {}
        args_match = re.search(r'"arguments"\s*:\s*\{([^}]*)\}', body)
        if args_match:
            for kv in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', args_match.group(1)):
                args[kv.group(1)] = kv.group(2)
        return {"name": name_match.group(1), "arguments": args}
    return None


def parse_tool_call(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if "<tool_call>" in text:
        body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        result = _parse_one_tool_block(body)
        if result:
            return result

    clean = text.replace("```json", "").replace("```", "").strip()
    for candidate in [clean] + [m.group(0) for m in re.finditer(r"\{[\s\S]*?\}", text)]:
        try:
            d = json.loads(candidate)
            if isinstance(d, dict) and "name" in d:
                # Require "arguments" or "parameters" key to distinguish
                # tool call JSON from data objects that happen to have a
                # "name" field (e.g. employee records, PII data).
                if "arguments" not in d and "parameters" not in d:
                    continue
                d.setdefault("arguments", d.pop("parameters", {}))
                if not isinstance(d.get("arguments"), dict):
                    d["arguments"] = {}
                return d
        except Exception:
            continue
    return None


def parse_all_tool_calls(text: str) -> list:
    """Extract ALL tool calls from text that may contain multiple <tool_call> blocks."""
    text = (text or "").strip()
    results = []

    # Extract all <tool_call>...</tool_call> blocks
    parts = text.split("<tool_call>")
    if len(parts) > 1:
        for part in parts[1:]:
            body = part.split("</tool_call>", 1)[0].strip()
            parsed = _parse_one_tool_block(body)
            if parsed:
                results.append(parsed)
        if results:
            return results

    # Fallback: single tool call via existing parser
    single = parse_tool_call(text)
    if single:
        return [single]
    return []


def classify_output(text: str) -> dict:
    parsed = parse_tool_call(text)
    if parsed:
        return {"type": "tool", "tool_name": parsed["name"],
                "arguments": parsed.get("arguments", {}), "raw_text": text}
    return {"type": "content", "content": text, "raw_text": text}


def classify_output_multi(text: str, model_family: str = "qwen", processor=None) -> dict:
    """Like classify_output but returns all tool calls when multiple are present."""
    if model_family == "gemma":
        gemma_calls = parse_all_gemma_tool_calls(text, processor=processor)
        if gemma_calls:
            return {"type": "tool", "tool_calls": gemma_calls, "raw_text": text}
    if model_family == "qwen3.5":
        qwen35_calls = parse_qwen35_tool_calls(text)
        if qwen35_calls:
            return {"type": "tool", "tool_calls": qwen35_calls, "raw_text": text}
    all_calls = parse_all_tool_calls(text)
    if all_calls:
        return {"type": "tool", "tool_calls": all_calls, "raw_text": text}
    return {"type": "content", "content": text, "raw_text": text}


def apply_chat_template(tokenizer, messages, tools=None, model_family: str = "qwen"):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if tools:
        kwargs["tools"] = tools
    if model_family != "gemma":
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_generation_config(model_path: str) -> dict:
    """Load generation config from model, return only non-None params."""
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_path)
    except OSError:
        print(f"  No generation_config.json found for {model_path}, using defaults")
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

    params = {}
    temperature = getattr(gen_config, "temperature", None)
    if temperature is not None:
        params["temperature"] = float(temperature)

    top_p = getattr(gen_config, "top_p", None)
    if top_p is not None:
        params["top_p"] = float(top_p)

    top_k = getattr(gen_config, "top_k", None)
    if top_k is not None and top_k > 0:
        params["top_k"] = int(top_k)

    return params


# ---------------------------------------------------------------------------
# Task loading: local JSON or HuggingFace
# ---------------------------------------------------------------------------
HF_DATASET_ID = "cesun/SODA"

_PATH_TO_HF_CONFIG = {
    "tasks_full_interaction.json": "full_interaction",
    "tasks_compliant_response.json": "compliant_response",
    "tasks_random_response.json": "random_response",
    "tasks_empty_response.json": "empty_response",
    "tasks_random_request.json": "random_request",
    "tasks_empty_request.json": "empty_request",
    "tasks_all_random.json": "all_random",
    "tasks_all_empty.json": "all_empty",
}


def load_tasks(data_path: str) -> list:
    """Load tasks from local JSON, falling back to HuggingFace if not found.

    Automatically detects which HF config to use based on filename.
    """
    import os
    import json

    if os.path.exists(data_path):
        with open(data_path) as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {data_path}")
        return tasks

    basename = os.path.basename(data_path)
    if basename in _PATH_TO_HF_CONFIG:
        config = _PATH_TO_HF_CONFIG[basename]
        try:
            from datasets import load_dataset
            print(f"Loading from HuggingFace: {HF_DATASET_ID} ({config}/test)")
            ds = load_dataset(HF_DATASET_ID, config, split="test")
            tasks = []
            for row in ds:
                task = {
                    "id": row["id"],
                    "scenario": row["scenario"],
                    "threat_depth": row["threat_depth"],
                    "threat_source": row["threat_source"],
                    "harm_description": row["harm_description"],
                    "environments": [{
                        "name": row["env_name"],
                        "tools": json.loads(row["tools"]),
                        "parameters": json.loads(row["parameters"]),
                    }],
                    "user_turns": json.loads(row["user_turns"]),
                }
                task["prefilled_messages"] = json.loads(row["prefilled_messages"])
                tasks.append(task)
            print(f"Loaded {len(tasks)} tasks from HuggingFace")
            return tasks
        except ImportError:
            raise ImportError(
                "Local data not found. Install datasets to auto-download: pip install datasets"
            )

    raise FileNotFoundError(
        f"{data_path} not found. Run data generation scripts or install 'pip install datasets' "
        f"to auto-download from HuggingFace."
    )
