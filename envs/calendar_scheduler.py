from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class CalendarScheduler(BaseEnv):
    """Simulated calendar scheduling environment.

    State schema:
        events: list of dicts, each with keys:
            id, title, start, end, attendees, description, location
        contacts: list of dicts, each with keys:
            email, name
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("events", [])
        self.state.setdefault("contacts", [])

    def _get_event(self, event_id: str) -> Optional[Dict]:
        for ev in self.state["events"]:
            if ev["id"] == event_id:
                return ev
        return None

    def _next_event_id(self) -> str:
        return f"evt_{len(self.state['events']) + 1}"

    def get_events(
        self, event_id: str = "", start_date: str = "", end_date: str = "", query: str = ""
    ) -> Dict:
        # Direct lookup by event_id takes priority
        if event_id:
            ev = self._get_event(event_id)
            if ev is None:
                return {"success": False, "message": f"Event '{event_id}' not found."}
            result = {
                "id": ev["id"],
                "title": ev["title"],
                "start": ev.get("start", ""),
                "end": ev.get("end", ""),
                "location": ev.get("location", ""),
                "attendees": ev.get("attendees", []),
                "description": ev.get("description", ""),
            }
            return {"success": True, "events": [result], "total": 1}

        results = []
        for ev in self.state["events"]:
            if start_date and ev.get("start", "") < start_date:
                continue
            if end_date and ev.get("start", "") > end_date + "T23:59:59":
                continue
            if query and query.lower() not in ev.get("title", "").lower() and query.lower() not in ev.get("description", "").lower():
                continue
            results.append({
                "id": ev["id"],
                "title": ev["title"],
                "start": ev.get("start", ""),
                "end": ev.get("end", ""),
                "location": ev.get("location", ""),
                "attendees": ev.get("attendees", []),
                "description": ev.get("description", ""),
            })
        # Fallback: if filters produced no results but events exist, return
        # all events so the benchmark can reliably deliver threat content.
        if not results and self.state["events"]:
            for ev in self.state["events"]:
                results.append({
                    "id": ev["id"], "title": ev["title"],
                    "start": ev.get("start", ""), "end": ev.get("end", ""),
                    "location": ev.get("location", ""),
                    "attendees": ev.get("attendees", []),
                    "description": ev.get("description", ""),
                })
        return {"success": True, "events": results, "total": len(results)}

    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
        attendees: List[str] = None,
    ) -> Dict:
        event = {
            "id": self._next_event_id(),
            "title": title,
            "start": start,
            "end": end,
            "description": description,
            "location": location,
            "attendees": attendees or [],
        }
        self.state["events"].append(event)
        return {"success": True, "message": f"Event '{title}' created.", "event_id": event["id"]}

    def update_event(
        self,
        event_id: str,
        title: str = "",
        start: str = "",
        end: str = "",
        description: str = "",
        location: str = "",
    ) -> Dict:
        event = self._get_event(event_id)
        if event is None:
            return {"success": False, "message": f"Event '{event_id}' not found."}
        if title:
            event["title"] = title
        if start:
            event["start"] = start
        if end:
            event["end"] = end
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        return {"success": True, "message": f"Event '{event_id}' updated."}

    def delete_event(self, event_id: str) -> Dict:
        event = self._get_event(event_id)
        if event is None:
            return {"success": False, "message": f"Event '{event_id}' not found."}
        self.state["events"] = [e for e in self.state["events"] if e["id"] != event_id]
        return {"success": True, "message": f"Event '{event_id}' deleted."}

    def invite_participant(self, event_id: str, email: str, message: str = "") -> Dict:
        event = self._get_event(event_id)
        if event is None:
            return {"success": False, "message": f"Event '{event_id}' not found."}
        attendees = event.get("attendees", [])
        if email in attendees:
            return {"success": False, "message": f"'{email}' is already an attendee."}
        attendees.append(email)
        event["attendees"] = attendees
        return {"success": True, "message": f"Invitation sent to '{email}' for event '{event['title']}'."}

    def check_availability(self, email: str, start: str, end: str) -> Dict:
        conflicts = []
        for ev in self.state["events"]:
            if email in ev.get("attendees", []):
                if ev.get("start", "") < end and ev.get("end", "") > start:
                    conflicts.append({
                        "id": ev["id"],
                        "title": ev["title"],
                        "start": ev["start"],
                        "end": ev["end"],
                    })
        available = len(conflicts) == 0
        return {"success": True, "available": available, "conflicts": conflicts}
