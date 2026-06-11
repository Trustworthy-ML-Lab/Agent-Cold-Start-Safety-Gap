#!/usr/bin/env python3
"""Convert API-Bank API classes to OpenAI tool schema format.

Reads all API classes from apis/ and outputs a JSON file with tool schemas
compatible with OpenAI function-calling format.

Usage:
    python external_eval/api_bank/convert_to_tool_schema.py
"""
import importlib
import json
import os
import sys
from pathlib import Path

API_DIR = Path(__file__).parent / "apis"
sys.path.insert(0, str(Path(__file__).parent))

TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
}


def convert_api_to_tool_schema(api_class) -> dict:
    """Convert an API class to OpenAI tool schema format."""
    name = api_class.__name__
    description = getattr(api_class, "description", f"API: {name}")
    input_params = getattr(api_class, "input_parameters", {})

    properties = {}
    required = []
    for param_name, param_info in input_params.items():
        param_type = TYPE_MAP.get(param_info.get("type", "str"), "string")
        prop = {
            "type": param_type,
            "description": param_info.get("description", ""),
        }
        properties[param_name] = prop
        required.append(param_name)

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    return schema


def load_all_apis():
    """Load all API classes from the apis directory."""
    api_classes = {}
    for fname in sorted(os.listdir(API_DIR)):
        if not fname.endswith(".py") or fname.startswith("_") or fname == "api.py":
            continue
        module_name = fname[:-3]
        try:
            mod = importlib.import_module(f"apis.{module_name}")
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (isinstance(obj, type) and
                    hasattr(obj, "input_parameters") and
                    hasattr(obj, "description") and
                    attr_name != "API"):
                    api_classes[attr_name] = obj
        except Exception as e:
            print(f"  Warning: failed to load {module_name}: {e}")
    return api_classes


def main():
    print("Loading API classes...")
    api_classes = load_all_apis()
    print(f"  Found {len(api_classes)} APIs")

    # Convert to tool schemas
    tool_schemas = {}
    all_schemas = []
    for name, cls in sorted(api_classes.items()):
        schema = convert_api_to_tool_schema(cls)
        tool_schemas[name] = schema
        all_schemas.append(schema)
        print(f"  {name}: {len(schema['function']['parameters']['properties'])} params")

    # Save
    out_path = Path(__file__).parent.parent / "data" / "api_bank" / "tool_schemas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_schemas, f, indent=2)
    print(f"\nSaved {len(all_schemas)} tool schemas to {out_path}")

    # Also save a mapping file (name -> schema) for per-task tool filtering
    map_path = Path(__file__).parent.parent / "data" / "api_bank" / "tool_schemas_by_name.json"
    with open(map_path, "w") as f:
        json.dump(tool_schemas, f, indent=2)
    print(f"Saved name->schema mapping to {map_path}")


if __name__ == "__main__":
    main()
