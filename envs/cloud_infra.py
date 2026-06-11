from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class CloudInfra(BaseEnv):
    """Simulated cloud infrastructure management environment.

    State schema:
        instances: list of dicts, each with keys:
            id, name, status, region, type, ip, security_groups
        security_groups: list of dicts, each with keys:
            id, name, rules (list of rule dicts with protocol, port, cidr, direction)
        snapshots: list of dicts, each with keys:
            id, instance_id, name, created_at
        services: list of dicts, each with keys:
            name, instance_id, image, status, env_vars
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("instances", [])
        self.state.setdefault("security_groups", [])
        self.state.setdefault("snapshots", [])
        self.state.setdefault("services", [])

        # Normalise field names: templates use "state" for instance status,
        # but the env code expects "status".
        for inst in self.state["instances"]:
            if "state" in inst and "status" not in inst:
                inst["status"] = inst.pop("state")
            elif "state" in inst and "status" in inst:
                # prefer "state" from template if both present
                inst["status"] = inst.pop("state")

        # Some templates load security-group-shaped items (id starts with
        # "sg-" or has "rules" key) into the "instances" list because the
        # template state_key is always "instances".  Copy them to
        # security_groups so modify_security_group can find them, but ALSO
        # keep them in the instances list so that list_instances still
        # returns their data (including embedded threat payloads).
        for item in self.state["instances"]:
            item_id = item.get("id", "")
            if (isinstance(item_id, str) and item_id.startswith("sg-")) or "rules" in item:
                self.state["security_groups"].append(item)

        # Auto-generate a security group for each instance that references
        # security groups (via "security_groups" list field) or that simply
        # doesn't have one yet.  This ensures security_review scenarios can
        # discover groups via list_security_groups.
        existing_sg_ids = {sg["id"] for sg in self.state["security_groups"]}
        for inst in self.state["instances"]:
            inst_id = inst.get("id", "")
            inst_name = inst.get("name", inst_id)
            # If the instance already has associated sg refs, create stubs
            # for any that don't exist yet.
            sg_refs = inst.get("security_groups", [])
            if isinstance(sg_refs, list):
                for sg_ref in sg_refs:
                    sg_id = sg_ref if isinstance(sg_ref, str) else sg_ref.get("id", "")
                    if sg_id and sg_id not in existing_sg_ids:
                        self.state["security_groups"].append({
                            "id": sg_id,
                            "name": sg_id,
                            "rules": [],
                        })
                        existing_sg_ids.add(sg_id)
            # Also generate a default security group for each instance so
            # that security_review filler turns can find something.
            auto_sg_id = f"sg-{inst_id}" if not inst_id.startswith("sg-") else inst_id
            if auto_sg_id not in existing_sg_ids:
                self.state["security_groups"].append({
                    "id": auto_sg_id,
                    "name": f"{inst_name}-sg",
                    "rules": [
                        {"protocol": "tcp", "port": 22, "cidr": "10.0.0.0/8", "direction": "inbound"},
                        {"protocol": "tcp", "port": 443, "cidr": "0.0.0.0/0", "direction": "inbound"},
                    ],
                })
                existing_sg_ids.add(auto_sg_id)

    def _get_instance(self, instance_id: str) -> Optional[Dict]:
        for inst in self.state["instances"]:
            if inst["id"] == instance_id:
                return inst
        # Fallback: also search security_groups so that sg-* items loaded
        # as data threats can be discovered via instance-oriented queries.
        for sg in self.state["security_groups"]:
            if sg["id"] == instance_id:
                return sg
        return None

    def _get_security_group(self, group_id: str) -> Optional[Dict]:
        for sg in self.state["security_groups"]:
            if sg["id"] == group_id:
                return sg
        return None

    def list_instances(self, status: str = "", region: str = "") -> Dict:
        results = []
        for inst in self.state["instances"]:
            inst_status = inst.get("status", inst.get("state", ""))
            if status and inst_status != status:
                continue
            if region:
                inst_region = inst.get("region", "")
                # Support prefix matching so that "eu" matches "eu-west-1",
                # "us-east" matches "us-east-1", etc.  Also handle
                # segment-level substring: "eu" as a complete segment within
                # the region identifier.
                region_lower = region.lower().rstrip("-")
                inst_region_lower = inst_region.lower()
                # Split region into segments by "-" for segment matching
                inst_segments = inst_region_lower.split("-")
                match_region = (
                    inst_region_lower == region_lower
                    or inst_region_lower.startswith(region_lower + "-")
                    or region_lower in inst_segments
                )
                if not match_region:
                    continue
            # Return all fields from the state item so that threat payloads
            # embedded in extra fields (env_vars, tags, snapshot_metadata,
            # app_config, security_scan_results, etc.) are delivered.
            entry = dict(inst)
            # Ensure canonical "status" key is always present
            entry.setdefault("status", inst_status or "unknown")
            # Attach the auto-generated or referenced security group IDs
            # so the model can discover which groups to inspect.
            if "security_groups" not in entry:
                auto_sg_id = f"sg-{inst.get('id', '')}"
                sgs = [sg["id"] for sg in self.state["security_groups"]
                       if sg["id"] == auto_sg_id]
                if sgs:
                    entry["security_groups"] = sgs
            results.append(entry)
        return {"success": True, "instances": results, "total": len(results)}

    def start_instance(self, instance_id: str) -> Dict:
        inst = self._get_instance(instance_id)
        if inst is None:
            return {"success": False, "message": f"Instance '{instance_id}' not found."}
        if inst.get("status") == "running":
            return {"success": False, "message": f"Instance '{instance_id}' is already running."}
        if inst.get("status") == "terminated":
            return {"success": False, "message": f"Instance '{instance_id}' is terminated and cannot be started."}
        inst["status"] = "running"
        return {"success": True, "message": f"Instance '{instance_id}' started."}

    def stop_instance(self, instance_id: str, force: bool = False) -> Dict:
        inst = self._get_instance(instance_id)
        if inst is None:
            return {"success": False, "message": f"Instance '{instance_id}' not found."}
        if inst.get("status") == "stopped":
            return {"success": False, "message": f"Instance '{instance_id}' is already stopped."}
        if inst.get("status") == "terminated":
            return {"success": False, "message": f"Instance '{instance_id}' is terminated."}
        inst["status"] = "stopped"
        method = "force-stopped" if force else "gracefully stopped"
        return {"success": True, "message": f"Instance '{instance_id}' {method}."}

    def list_security_groups(self) -> Dict:
        """List all security groups and their rules."""
        results = []
        for sg in self.state["security_groups"]:
            entry = dict(sg)
            entry.setdefault("rules", [])
            results.append(entry)
        return {"success": True, "security_groups": results, "total": len(results)}

    def modify_security_group(
        self, group_id: str, action: str, protocol: str, port: int, cidr: str
    ) -> Dict:
        sg = self._get_security_group(group_id)
        if sg is None:
            return {"success": False, "message": f"Security group '{group_id}' not found."}

        rule = {"protocol": protocol, "port": port, "cidr": cidr}
        rules = sg.setdefault("rules", [])

        if action == "add":
            for existing in rules:
                if (existing.get("protocol") == protocol
                        and existing.get("port") == port
                        and existing.get("cidr") == cidr):
                    return {"success": False, "message": "Rule already exists."}
            rules.append(rule)
            return {"success": True, "message": f"Rule added to security group '{group_id}'."}
        elif action == "remove":
            original_len = len(rules)
            sg["rules"] = [
                r for r in rules
                if not (r.get("protocol") == protocol and r.get("port") == port and r.get("cidr") == cidr)
            ]
            if len(sg["rules"]) == original_len:
                return {"success": False, "message": "Matching rule not found."}
            return {"success": True, "message": f"Rule removed from security group '{group_id}'."}
        else:
            return {"success": False, "message": f"Invalid action '{action}'. Must be 'add' or 'remove'."}

    def create_snapshot(self, instance_id: str, name: str) -> Dict:
        inst = self._get_instance(instance_id)
        if inst is None:
            return {"success": False, "message": f"Instance '{instance_id}' not found."}
        snapshot_id = f"snap_{len(self.state['snapshots']) + 1}"
        snapshot = {
            "id": snapshot_id,
            "instance_id": instance_id,
            "name": name,
            "created_at": "2025-01-01T00:00:00Z",
        }
        self.state["snapshots"].append(snapshot)
        return {"success": True, "message": f"Snapshot '{name}' created.", "snapshot_id": snapshot_id}

    def deploy_service(
        self, service_name: str, instance_id: str, image: str, env_vars: Dict = None
    ) -> Dict:
        inst = self._get_instance(instance_id)
        if inst is None:
            return {"success": False, "message": f"Instance '{instance_id}' not found."}
        if inst.get("status") != "running":
            return {"success": False, "message": f"Instance '{instance_id}' is not running. Start it first."}

        # Check if service already exists and update, otherwise create
        for svc in self.state["services"]:
            if svc["name"] == service_name and svc["instance_id"] == instance_id:
                svc["image"] = image
                svc["env_vars"] = env_vars or {}
                svc["status"] = "deployed"
                return {"success": True, "message": f"Service '{service_name}' redeployed on '{instance_id}'."}

        service = {
            "name": service_name,
            "instance_id": instance_id,
            "image": image,
            "status": "deployed",
            "env_vars": env_vars or {},
        }
        self.state["services"].append(service)
        return {"success": True, "message": f"Service '{service_name}' deployed on '{instance_id}'."}
