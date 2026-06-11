"""External eval utilities — tool call parsing, env loading, tool execution.

Parsing logic copied from src/utils.py to keep external_eval self-contained.
"""
import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure ASB environments are importable
ASB_ENVS_PATH = str(Path(__file__).resolve().parent.parent / "asb" / "environments")
if ASB_ENVS_PATH not in sys.path:
    sys.path.insert(0, ASB_ENVS_PATH)


def is_native_tool_format(model_path: str) -> bool:
    return "llama" in model_path.strip().lower()


# ---------------------------------------------------------------------------
# Gemma tool call parsing
# ---------------------------------------------------------------------------

def _parse_gemma_args(raw: str) -> dict:
    args = {}
    raw = raw.replace('<|"|>', '"')
    # Try JSON first
    try:
        d = json.loads("{" + raw + "}")
        return d
    except Exception:
        pass
    # Try adding quotes around unquoted values
    try:
        # Pattern: key:value,key:value — add quotes around string values
        quoted = re.sub(r'(\w+)\s*:\s*(?![\d\[\{"\'-])(.*?)(?=,\s*\w+\s*:|$)',
                       lambda m: f'"{m.group(1)}": "{m.group(2).strip()}"', raw)
        d = json.loads("{" + quoted + "}")
        return d
    except Exception:
        pass
    # Manual parsing: split by comma-then-key pattern
    # Match key:value pairs where value extends until next ,key: or end
    pairs = re.split(r',(?=\w+:)', raw)
    for pair in pairs:
        pair = pair.strip()
        if ':' not in pair:
            continue
        key, val = pair.split(':', 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # Try to interpret the value
        if val.lower() in ('true', 'false'):
            args[key] = val.lower() == 'true'
        elif val.startswith('[') and val.endswith(']'):
            try:
                args[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                inner = val[1:-1].strip()
                if inner:
                    args[key] = [item.strip() for item in inner.split(',')]
                else:
                    args[key] = []
        else:
            try:
                args[key] = int(val)
            except ValueError:
                try:
                    args[key] = float(val)
                except ValueError:
                    args[key] = val
    return args


def parse_all_gemma_tool_calls(text: str, processor=None) -> list:
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
    for match in re.finditer(r'<\|tool_call>call:([\w.]+)\{(.*?)\}<tool_call\|>', text, re.DOTALL):
        name = match.group(1)
        args = _parse_gemma_args(match.group(2).strip())
        results.append({"name": name, "arguments": args})
    if not results:
        for match in re.finditer(r'call:([\w.]+)\{(.*?)\}', text, re.DOTALL):
            name = match.group(1)
            args = _parse_gemma_args(match.group(2).strip())
            results.append({"name": name, "arguments": args})
    return results


# ---------------------------------------------------------------------------
# Qwen 3.5 XML-style tool call parsing
# ---------------------------------------------------------------------------

def parse_qwen35_tool_calls(text: str) -> list:
    results = []
    blocks = text.split("<tool_call>")
    for block in blocks[1:]:
        body = block.split("</tool_call>", 1)[0].strip()
        fn_match = re.search(r'<function=([\w.]+)>(.*?)</function>', body, re.DOTALL)
        if not fn_match:
            continue
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

    if not results:
        for fn_match in re.finditer(r'<function=([\w.]+)>(.*?)</function>', text, re.DOTALL):
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


# ---------------------------------------------------------------------------
# Standard JSON tool call parsing
# ---------------------------------------------------------------------------

def _parse_one_tool_block(body: str) -> Optional[dict]:
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
    text = (text or "").strip()
    results = []

    # Try <tool_call>...</tool_call> blocks first
    parts = text.split("<tool_call>")
    if len(parts) > 1:
        for part in parts[1:]:
            body = part.split("</tool_call>", 1)[0].strip()
            parsed = _parse_one_tool_block(body)
            if parsed:
                results.append(parsed)
        if results:
            return results

    # Try multiple JSON objects separated by newline, semicolon, or }{
    # Split by common separators between tool calls
    parts = re.split(r'\}\s*[;\n]\s*\{', text)
    if len(parts) > 1:
        for i, part in enumerate(parts):
            if i > 0:
                part = '{' + part
            if i < len(parts) - 1:
                part = part + '}'
            parsed = _parse_one_tool_block(part)
            if parsed:
                results.append(parsed)
        if results:
            return results

    # Try splitting by newline only
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('{') and '"name"' in line:
            parsed = _parse_one_tool_block(line)
            if parsed:
                results.append(parsed)
    if results:
        return results

    # Fallback: single tool call
    single = parse_tool_call(text)
    if single:
        return [single]
    return []


# ---------------------------------------------------------------------------
# classify_output — same as src/utils.py
# ---------------------------------------------------------------------------

def classify_output(text: str) -> dict:
    parsed = parse_tool_call(text)
    if parsed:
        return {"type": "tool", "tool_name": parsed["name"],
                "arguments": parsed.get("arguments", {}), "raw_text": text}
    # Try Qwen 3.5 XML format
    qwen35_calls = parse_qwen35_tool_calls(text)
    if qwen35_calls:
        return {"type": "tool", "tool_name": qwen35_calls[0]["name"],
                "arguments": qwen35_calls[0].get("arguments", {}), "raw_text": text}
    # Try Gemma call:func{args} format
    gemma_calls = parse_all_gemma_tool_calls(text)
    if gemma_calls:
        return {"type": "tool", "tool_name": gemma_calls[0]["name"],
                "arguments": gemma_calls[0].get("arguments", {}), "raw_text": text}
    return {"type": "content", "content": text, "raw_text": text}


def classify_output_multi(text: str, model_family: str = "qwen", processor=None) -> dict:
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


# ---------------------------------------------------------------------------
# ASB environment loading and tool execution
# ---------------------------------------------------------------------------

def load_env_tools(task: dict) -> Tuple[dict, list]:
    envs_map, tools = {}, []
    for cfg in task.get("environments", []):
        name = cfg.get("name", "")
        if not name:
            continue
        try:
            cls = getattr(importlib.import_module(name), name)
            env = cls(parameters=cfg.get("parameters") or None)
            for desc in env.get_tool_descs(cfg.get("tools", []) or env.tool_list):
                tools.append({"type": "function", "function": desc})
                envs_map[desc["name"]] = env
        except Exception:
            pass
    return envs_map, tools


def call_tool(envs_map: dict, name: str, arguments: dict) -> dict:
    if name not in envs_map:
        return {"success": False, "message": f"Tool {name} not available."}
    try:
        result = envs_map[name].call_tool(name, arguments)
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        return {"success": False, "message": str(e)}
