from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class SmartHome(BaseEnv):
    """Simulated smart home environment.

    State schema:
        devices: list of dicts, each with keys:
            id, type, name, status (dict of state values), location
        schedules: list of dicts, each with keys:
            id, device_id, schedule_name, time, days, action
        camera_feeds: dict mapping camera_id -> feed info dict
            with keys: snapshot_url, stream_url, last_motion
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("devices", [])
        self.state.setdefault("schedules", [])
        self.state.setdefault("camera_feeds", {})

        # Deduplicate devices by ID, keeping the LAST occurrence.
        # Threat items are appended after filler items and may reuse
        # a filler ID; the threat version must take priority so its
        # embedded data is discoverable.
        seen: Dict[str, int] = {}
        deduped: List[Dict] = []
        for dev in self.state["devices"]:
            dev_id = dev.get("id", "")
            if dev_id in seen:
                # Replace the earlier entry with the newer one.
                deduped[seen[dev_id]] = dev
            else:
                seen[dev_id] = len(deduped)
                deduped.append(dev)
        self.state["devices"] = deduped

    def _get_device(self, device_id: str) -> Optional[Dict]:
        # Exact match first
        for dev in self.state["devices"]:
            if dev["id"] == device_id:
                return dev
        # Fallback: try matching by name (case-insensitive)
        device_id_lower = device_id.lower().replace("_", " ").replace("-", " ")
        for dev in self.state["devices"]:
            name = dev.get("name", "").lower().replace("_", " ").replace("-", " ")
            if name and (device_id_lower in name or name in device_id_lower):
                return dev
        # Fallback: try matching device_id substring against id or type+location
        for dev in self.state["devices"]:
            dev_type = dev.get("type", "").lower()
            dev_loc = dev.get("location", "").lower()
            # e.g. "thermostat_office" matches a device with type=thermostat, location=office
            if dev_type and dev_loc and dev_type in device_id_lower and dev_loc in device_id_lower:
                return dev
        return None

    def list_devices(self, device_type: str = "", location: str = "") -> Dict:
        results = []
        for dev in self.state["devices"]:
            if device_type and dev.get("type", "") != device_type:
                continue
            if location and dev.get("location", "") != location:
                continue
            results.append({
                "id": dev["id"],
                "type": dev.get("type", ""),
                "name": dev.get("name", ""),
                "location": dev.get("location", ""),
                "status": dev.get("status", {}),
            })
        return {"success": True, "devices": results, "total": len(results)}

    def set_device_state(self, device_id: str, state: Dict) -> Dict:
        dev = self._get_device(device_id)
        if dev is None:
            return {"success": False, "message": f"Device '{device_id}' not found."}
        current_status = dev.setdefault("status", {})
        for k, v in state.items():
            current_status[k] = v
        return {
            "success": True,
            "message": f"Device '{dev.get('name', device_id)}' state updated.",
            "new_status": current_status,
        }

    def get_device_status(self, device_id: str) -> Dict:
        dev = self._get_device(device_id)
        if dev is None:
            return {"success": False, "message": f"Device '{device_id}' not found."}
        return {
            "success": True,
            "device": {
                "id": dev["id"],
                "type": dev.get("type", ""),
                "name": dev.get("name", ""),
                "location": dev.get("location", ""),
                "status": dev.get("status", {}),
            },
        }

    def set_schedule(
        self,
        device_id: str,
        schedule_name: str,
        time: str,
        action: Dict,
        days: List[str] = None,
    ) -> Dict:
        dev = self._get_device(device_id)
        if dev is None:
            return {"success": False, "message": f"Device '{device_id}' not found."}

        schedule_id = f"sched_{len(self.state['schedules']) + 1}"
        schedule = {
            "id": schedule_id,
            "device_id": device_id,
            "schedule_name": schedule_name,
            "time": time,
            "days": days or ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            "action": action,
        }
        self.state["schedules"].append(schedule)
        return {
            "success": True,
            "message": f"Schedule '{schedule_name}' created for device '{dev.get('name', device_id)}'.",
            "schedule_id": schedule_id,
        }

    def get_camera_feed(self, camera_id: str, mode: str = "snapshot") -> Dict:
        dev = self._get_device(camera_id)
        if dev is None:
            return {"success": False, "message": f"Camera '{camera_id}' not found."}
        if dev.get("type") != "camera":
            return {"success": False, "message": f"Device '{camera_id}' is not a camera."}

        feed = self.state["camera_feeds"].get(camera_id, {})
        if mode == "snapshot":
            url = feed.get("snapshot_url", f"https://cameras.local/{camera_id}/snapshot.jpg")
            return {
                "success": True,
                "camera_id": camera_id,
                "mode": "snapshot",
                "url": url,
                "last_motion": feed.get("last_motion", ""),
            }
        elif mode == "stream":
            url = feed.get("stream_url", f"rtsp://cameras.local/{camera_id}/live")
            return {
                "success": True,
                "camera_id": camera_id,
                "mode": "stream",
                "url": url,
            }
        else:
            return {"success": False, "message": f"Invalid mode '{mode}'. Must be 'snapshot' or 'stream'."}

    def unlock_door(self, lock_id: str, duration: int = 30) -> Dict:
        dev = self._get_device(lock_id)
        if dev is None:
            return {"success": False, "message": f"Lock '{lock_id}' not found."}
        if dev.get("type") != "lock":
            return {"success": False, "message": f"Device '{lock_id}' is not a lock."}

        dev.setdefault("status", {})["locked"] = False
        return {
            "success": True,
            "message": f"Door '{dev.get('name', lock_id)}' unlocked for {duration} seconds.",
            "lock_id": lock_id,
            "auto_lock_seconds": duration,
        }
