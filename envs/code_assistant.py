from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class CodeAssistant(BaseEnv):
    """Simulated code assistant environment.

    State schema:
        files: dict mapping path -> content (source code files)
        packages: list of dicts, each with keys:
            name, version
        deployments: list of dicts, each with keys:
            id, path, target, status
        docs: list of dicts, each with keys:
            library, topic, content
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("files", {})
        self.state.setdefault("packages", [])
        self.state.setdefault("deployments", [])
        self.state.setdefault("docs", [])
        if isinstance(self.state["files"], list):
            # Convert list-of-dicts to path->content dict.
            # When duplicate paths exist, keep all entries by appending
            # a numeric suffix so no content is lost.
            files_dict: Dict[str, str] = {}
            for f in self.state["files"]:
                if not isinstance(f, dict) or "path" not in f:
                    continue
                path = f["path"]
                content = f.get("content", "")
                if path not in files_dict:
                    files_dict[path] = content
                else:
                    # Duplicate path: keep as alternate entry only if
                    # the content is actually different (avoids clutter
                    # from identical filler items).
                    if files_dict[path] != content:
                        idx = 2
                        while f"{path}.v{idx}" in files_dict:
                            idx += 1
                        files_dict[f"{path}.v{idx}"] = content
            self.state["files"] = files_dict

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a file path, trying exact match first then basename fallback."""
        if path in self.state["files"]:
            return path
        # Try stripping common prefixes the model may prepend
        stripped = path.lstrip("/")
        for prefix in ("project_root/", "repo/", "src/../", "./"):
            if stripped.startswith(prefix):
                candidate = stripped[len(prefix):]
                if candidate in self.state["files"]:
                    return candidate
        # Try matching by basename when only one file has that name
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        matches = [k for k in self.state["files"] if k.rsplit("/", 1)[-1] == basename]
        if len(matches) == 1:
            return matches[0]
        return None

    def read_code(self, path: str) -> Dict:
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"success": False, "message": f"File '{path}' not found."}
        content = self.state["files"][resolved]
        return {"success": True, "path": resolved, "content": content, "lines": content.count("\n") + 1}

    def write_code(self, path: str, content: str) -> Dict:
        is_new = path not in self.state["files"]
        self.state["files"][path] = content
        action = "created" if is_new else "updated"
        return {
            "success": True,
            "message": f"File '{path}' {action}.",
            "path": path,
            "lines": content.count("\n") + 1,
        }

    def run_code(self, path: str = "", snippet: str = "", language: str = "python") -> Dict:
        if not path and not snippet:
            return {"success": False, "message": "Provide either 'path' or 'snippet' to execute."}
        if path:
            resolved = self._resolve_path(path)
            if resolved is None:
                return {"success": False, "message": f"File '{path}' not found."}
            content = self.state["files"][resolved]
            label = resolved
        else:
            content = snippet
            label = f"<{language} snippet>"
        return {
            "success": True,
            "message": f"Executed {label}.",
            "stdout": f"[simulated output of {label}]",
            "stderr": "",
            "exit_code": 0,
        }

    def install_package(self, name: str, version: str = "") -> Dict:
        existing = next((p for p in self.state["packages"] if p["name"] == name), None)
        resolved_version = version if version else "latest"
        if existing:
            existing["version"] = resolved_version
            return {
                "success": True,
                "message": f"Package '{name}' updated to {resolved_version}.",
            }
        self.state["packages"].append({"name": name, "version": resolved_version})
        return {
            "success": True,
            "message": f"Package '{name}' ({resolved_version}) installed.",
        }

    def search_docs(self, query: str, library: str = "") -> Dict:
        results: List[Dict] = []
        for doc in self.state["docs"]:
            match = True
            if library and doc.get("library", "").lower() != library.lower():
                match = False
            if query.lower() not in doc.get("topic", "").lower() and query.lower() not in doc.get("content", "").lower():
                match = False
            if match:
                results.append({
                    "library": doc.get("library", ""),
                    "topic": doc.get("topic", ""),
                    "snippet": doc.get("content", "")[:300],
                })
        return {"success": True, "results": results, "total": len(results)}

    def deploy_app(self, path: str, target: str = "staging", config: dict = None) -> Dict:
        # Treat ".", "./", and "/" as project root matching any file
        is_root = path in (".", "./", "/", "")
        if not is_root and path not in self.state["files"] and not any(
            k.startswith(path.rstrip("/") + "/") for k in self.state["files"]
        ):
            return {"success": False, "message": f"Application path '{path}' not found."}
        if target not in ("staging", "production", "local"):
            target = "staging"
        deployment = {
            "id": f"deploy_{len(self.state['deployments']) + 1}",
            "path": path,
            "target": target,
            "status": "deployed",
            "config": config or {},
        }
        self.state["deployments"].append(deployment)
        return {
            "success": True,
            "message": f"Application deployed to {target}.",
            "deployment_id": deployment["id"],
            "target": target,
        }
