from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class FileSystem(BaseEnv):
    """Simulated file system environment.

    State schema:
        files: list of dicts, each with keys:
            path, content, permissions (e.g. "rwx", "r--", "rw-")
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("files", [])

    def _find_file(self, path: str) -> Optional[Dict]:
        for f in self.state["files"]:
            if f["path"] == path:
                return f
        return None

    def _is_dir(self, path: str) -> bool:
        prefix = path.rstrip("/") + "/"
        return any(f["path"].startswith(prefix) for f in self.state["files"])

    def _resolve_path(self, path: str) -> str:
        """Resolve a possibly-relative path against known file system entries.

        If *path* is already absolute (starts with ``/``) it is returned
        as-is.  Otherwise the method searches for an existing directory or
        file whose path ends with ``/<path>`` (first match wins) so that
        relative names like ``tests`` or ``src`` are correctly resolved to
        their absolute locations (e.g. ``/project/tests``).
        """
        if path.startswith("/"):
            return path
        suffix = "/" + path.strip("/")
        for f in self.state["files"]:
            if f["path"].endswith(suffix) or f["path"] == path:
                return f["path"]
            if f["path"].startswith(path.rstrip("/") + "/"):
                # There is a child under this relative name; derive
                # the absolute parent from the child path.
                idx = f["path"].find(suffix + "/")
                if idx != -1:
                    return f["path"][:idx + len(suffix)]
        # Fallback: return as-is (caller will find no matches).
        return path

    def list_files(self, path: str = "/", recursive: bool = False) -> Dict:
        path = self._resolve_path(path)
        prefix = path.rstrip("/") + "/"
        results: List[Dict] = []
        for f in self.state["files"]:
            if not f["path"].startswith(prefix) and f["path"] != path:
                continue
            rel = f["path"][len(prefix):]
            if not recursive and "/" in rel:
                continue
            results.append({
                "path": f["path"],
                "size": len(f.get("content", "")),
                "permissions": f.get("permissions", "rw-"),
            })
        return {"success": True, "files": results, "total": len(results)}

    def read_file(self, path: str) -> Dict:
        f = self._find_file(path)
        if f is None:
            return {"success": False, "message": f"File '{path}' not found."}
        perms = f.get("permissions", "rw-")
        if "r" not in perms:
            return {"success": False, "message": f"Permission denied: cannot read '{path}'."}
        return {"success": True, "path": path, "content": f.get("content", "")}

    def write_file(self, path: str, content: str) -> Dict:
        f = self._find_file(path)
        if f is not None:
            perms = f.get("permissions", "rw-")
            if "w" not in perms:
                return {"success": False, "message": f"Permission denied: cannot write to '{path}'."}
            f["content"] = content
            return {"success": True, "message": f"File '{path}' updated.", "bytes_written": len(content)}
        new_file = {"path": path, "content": content, "permissions": "rw-"}
        self.state["files"].append(new_file)
        return {"success": True, "message": f"File '{path}' created.", "bytes_written": len(content)}

    def delete_file(self, path: str) -> Dict:
        f = self._find_file(path)
        if f is None:
            if self._is_dir(path):
                prefix = path.rstrip("/") + "/"
                removed = [x for x in self.state["files"] if x["path"].startswith(prefix)]
                self.state["files"] = [x for x in self.state["files"] if not x["path"].startswith(prefix)]
                return {"success": True, "message": f"Directory '{path}' deleted.", "files_removed": len(removed)}
            return {"success": False, "message": f"File '{path}' not found."}
        self.state["files"].remove(f)
        return {"success": True, "message": f"File '{path}' deleted."}

    def copy_file(self, source: str, destination: str) -> Dict:
        f = self._find_file(source)
        if f is None:
            return {"success": False, "message": f"Source file '{source}' not found."}
        perms = f.get("permissions", "rw-")
        if "r" not in perms:
            return {"success": False, "message": f"Permission denied: cannot read '{source}'."}
        existing = self._find_file(destination)
        if existing is not None:
            existing["content"] = f.get("content", "")
            return {"success": True, "message": f"File copied to '{destination}' (overwritten)."}
        new_file = {"path": destination, "content": f.get("content", ""), "permissions": "rw-"}
        self.state["files"].append(new_file)
        return {"success": True, "message": f"File copied from '{source}' to '{destination}'."}

    def execute_script(self, path: str, args: str = "") -> Dict:
        f = self._find_file(path)
        # Simulated execution -- always succeed even if the script is not
        # in the file list (e.g. system binaries like /usr/bin/rm).
        # This lets the benchmark reliably test whether the model follows
        # through with executing harmful commands.
        content = f.get("content", "") if f is not None else ""
        return {
            "success": True,
            "message": f"Script '{path}' executed.",
            "stdout": f"[simulated output of {path} {args}]",
            "exit_code": 0,
        }
