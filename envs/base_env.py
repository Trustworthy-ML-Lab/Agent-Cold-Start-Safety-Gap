import copy
import json
import os
from typing import Any, Dict, List, Optional


class BaseEnv:
    """Base environment for agent safety benchmark.

    Each subclass represents a domain (email, bank, etc.) with configurable
    initial state. The same env class supports both harmful and benign scenarios
    by passing different state dicts at init time.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        class_name = self.__class__.__name__
        schema_path = os.path.join(os.path.dirname(__file__), f"{class_name}.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.tool_descs: List[Dict] = json.load(f)

        self.tool_list = [d["name"] for d in self.tool_descs]
        self.call_counts: Dict[str, int] = {name: 0 for name in self.tool_list}
        self.state: Dict[str, Any] = state or {}

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tool_list

    def get_tool_descs(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        if tool_names is None:
            return copy.deepcopy(self.tool_descs)
        out = []
        for tool_name in tool_names:
            for d in self.tool_descs:
                if d["name"] == tool_name:
                    out.append(copy.deepcopy(d))
                    break
            else:
                raise ValueError(f"Tool {tool_name} not found in {self.__class__.__name__}")
        return out

    def get_openai_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        descs = self.get_tool_descs(tool_names)
        return [{"type": "function", "function": d} for d in descs]

    def _coerce_args(self, tool_desc: Dict, arguments: Dict) -> tuple:
        required = tool_desc.get("parameters", {}).get("required", [])
        props = tool_desc.get("parameters", {}).get("properties", {})

        for req in required:
            if req not in arguments:
                return None, {"success": False, "message": f"Missing required parameter '{req}'."}

        args = copy.deepcopy(arguments)
        for key in list(args.keys()):
            if key not in props:
                del args[key]

        for key in args:
            ptype = props[key].get("type")
            try:
                if ptype == "integer":
                    args[key] = int(args[key])
                elif ptype == "number":
                    args[key] = float(args[key])
                elif ptype == "boolean":
                    args[key] = bool(args[key])
                elif ptype == "object" and not isinstance(args[key], dict):
                    args[key] = json.loads(args[key])
                elif ptype == "array" and not isinstance(args[key], list):
                    args[key] = json.loads(args[key])
            except Exception:
                return None, {"success": False, "message": f"Parameter '{key}' should be type '{ptype}'."}

        return args, None

    def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        if not hasattr(self, tool_name):
            return {"success": False, "message": f"Unknown tool '{tool_name}'."}

        tool_desc = self.get_tool_descs([tool_name])[0]
        safe_args, err = self._coerce_args(tool_desc, arguments)
        if err is not None:
            return err

        fn = getattr(self, tool_name)
        try:
            return fn(**safe_args)
        except Exception as e:
            return {"success": False, "message": f"Error: {type(e).__name__}: {str(e)[:200]}"}
        finally:
            self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
