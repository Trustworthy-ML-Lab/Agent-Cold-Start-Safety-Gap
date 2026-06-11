"""BFCL (Berkeley Function Calling Leaderboard) environment backbone.

Env logic:   envs/bfcl/environments/ (12 API classes, self-contained Python)
Func docs:   envs/bfcl/func_docs/ (tool schemas per API class)
Questions:   data/bfcl/BFCL_v4_multi_turn_*.json (600 multi-turn tasks)

12 API classes:
  GorillaFileSystem, VehicleControlAPI, TradingBot, TravelAPI,
  TicketAPI, MessageAPI, TwitterAPI (PostingAPI), MathAPI,
  MemoryAPI_kv, MemoryAPI_rec_sum, MemoryAPI_vector, WebSearchAPI

3 task categories:
  base (200):       standard multi-turn with all tools available
  miss_func (200):  some tools removed → tests hallucinated tool calls
  miss_param (200): some params removed → tests wrong argument handling

Concepts from this benchmark:
  wrong_tool, wrong_args, hallucinated_tool, misinterpret_output
"""

import json
import sys
from pathlib import Path

ENVS_DIR = Path(__file__).parent / "environments"
DOCS_DIR = Path(__file__).parent / "func_docs"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bfcl"

if str(ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(ENVS_DIR))

# Map class names to module names
CLASS_TO_MODULE = {
    "GorillaFileSystem": "gorilla_file_system",
    "VehicleControlAPI": "vehicle_control",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TicketAPI": "ticket_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "MathAPI": "math_api",
    "MemoryAPI_kv": "memory_kv",
    "MemoryAPI_rec_sum": "memory_rec_sum",
    "MemoryAPI_vector": "memory_vector",
    "WebSearchAPI": "web_search",
}

# Map class names to the func_doc JSON filenames
CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system",
    "VehicleControlAPI": "vehicle_control",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TicketAPI": "ticket_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "MathAPI": "math_api",
    "MemoryAPI_kv": "memory_kv",
    "MemoryAPI_rec_sum": "memory_rec_sum",
    "MemoryAPI_vector": "memory_vector",
    "WebSearchAPI": "web_search",
}


def load_tasks(category: str = "base") -> list[dict]:
    """Load BFCL multi-turn tasks.

    Args:
        category: "base", "miss_func", or "miss_param"
    """
    path = DATA_DIR / f"BFCL_v4_multi_turn_{category}.json"
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_env(class_name: str, initial_config: dict | None = None):
    """Instantiate a BFCL API class with optional initial config.

    Args:
        class_name: e.g., "GorillaFileSystem", "VehicleControlAPI"
        initial_config: state initialization dict from task data
    """
    import importlib
    module_name = CLASS_TO_MODULE.get(class_name)
    if module_name is None:
        raise ValueError(f"Unknown class: {class_name}. Available: {list(CLASS_TO_MODULE.keys())}")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls()

    if initial_config and class_name in initial_config:
        if hasattr(instance, "_load_scenario"):
            instance._load_scenario(initial_config[class_name])

    return instance


def get_tool_schemas(class_name: str) -> list[dict]:
    """Get function schemas for a BFCL API class."""
    doc_name = CLASS_TO_DOC.get(class_name, class_name)
    doc_path = DOCS_DIR / f"{doc_name}.json"
    docs = []
    with open(doc_path) as f:
        content = f.read().strip()
        if content.startswith("["):
            docs = json.loads(content)
        else:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    return docs


def load_task_env(task: dict):
    """Load all environments + tool schemas for a BFCL task.

    Returns (env_map, all_schemas) where:
      env_map: {class_name: instance}
      all_schemas: list of function doc dicts
    """
    env_map = {}
    all_schemas = []
    config = task.get("initial_config", {})
    excluded_raw = task.get("excluded_function", [])
    excluded_set = set(excluded_raw) if isinstance(excluded_raw, list) else set()

    for class_name in task.get("involved_classes", []):
        env = load_env(class_name, config)
        env_map[class_name] = env

        try:
            schemas = get_tool_schemas(class_name)
            for s in schemas:
                func_name = s.get("name", "")
                if func_name not in excluded_set:
                    all_schemas.append(s)
        except Exception as e:
            print(f"Warning: cannot load schemas for {class_name}: {e}")

    return env_map, all_schemas


def call_tool(env_map: dict, class_name: str, method_name: str, arguments: dict) -> dict:
    """Execute a tool call on the appropriate BFCL environment.

    Args:
        env_map: {class_name: instance} from load_task_env
        class_name: which API class to call
        method_name: which method on that class
        arguments: keyword arguments
    """
    if class_name not in env_map:
        return {"error": f"Unknown class: {class_name}"}

    env = env_map[class_name]
    if not hasattr(env, method_name):
        return {"error": f"Method {method_name} not found on {class_name}"}

    try:
        func = getattr(env, method_name)
        result = func(**arguments)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
