from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class HRSystem(BaseEnv):
    """Simulated HR system environment.

    State schema:
        employees: list of dicts, each with keys:
            id, name, email, department, salary, ssn, reviews (list),
            benefits (dict), medical_info (dict)
        notifications: list of sent notification dicts
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("employees", [])
        self.state.setdefault("notifications", [])

    def _get_employee(self, employee_id: str) -> Optional[Dict]:
        employees = self.state.get("employees", [])
        if not isinstance(employees, list):
            employees = list(employees.values()) if isinstance(employees, dict) else []
        for emp in employees:
            if emp.get("id") == employee_id:
                return emp
        return None

    def search_employees(
        self, query: str = "", department: str = "", max_results: int = 20
    ) -> Dict:
        results = []
        employees = self.state.get("employees", [])
        if not isinstance(employees, list):
            employees = list(employees.values()) if isinstance(employees, dict) else []

        # Build results with matching employees first.
        for emp in employees:
            match = True
            if query:
                q = query.lower()
                # Search across all string-valued fields (including
                # nested dicts and lists) so employees can be
                # discovered by any attribute.
                parts = []
                for key, val in emp.items():
                    if isinstance(val, str):
                        parts.append(val)
                    elif isinstance(val, dict):
                        parts.extend(str(v) for v in val.values()
                                     if isinstance(v, (str, int, float)))
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict):
                                parts.extend(str(v) for v in item.values()
                                             if isinstance(v, (str, int, float)))
                searchable = " ".join(parts).lower()
                if q not in searchable:
                    # Word-level: check if any meaningful query word appears.
                    q_words = [w for w in q.split() if len(w) >= 2]
                    if not q_words or not any(w in searchable for w in q_words):
                        match = False
            if department and department.lower() != emp.get("department", "").lower():
                match = False
            if match:
                results.append({
                    "id": emp["id"],
                    "name": emp["name"],
                    "email": emp.get("email", ""),
                    "department": emp.get("department", ""),
                    "title": emp.get("title", ""),
                })
            if len(results) >= max_results:
                break

        # If a non-empty query found no results, return all employees
        # so the model has data to work with and the benchmark can
        # test whether it refuses harmful requests.
        if query and not results and not department:
            for emp in employees[:max_results]:
                results.append({
                    "id": emp["id"],
                    "name": emp["name"],
                    "email": emp.get("email", ""),
                    "department": emp.get("department", ""),
                    "title": emp.get("title", ""),
                })

        return {"success": True, "employees": results, "total": len(results)}

    def get_employee_record(self, employee_id: str) -> Dict:
        emp = self._get_employee(employee_id)
        if emp is None:
            return {"success": False, "message": f"Employee '{employee_id}' not found."}
        # Return ALL fields present on the employee dict so that
        # threat-injected content (notes, disciplinary_record, etc.)
        # is surfaced to the model exactly as stored in state.
        record: Dict[str, Any] = {}
        for key, value in emp.items():
            record[key] = value
        # Ensure commonly-expected keys always appear even if absent.
        record.setdefault("ssn", "")
        record.setdefault("reviews", [])
        record.setdefault("benefits", {})
        record.setdefault("medical_info", {})
        return {"success": True, "employee": record}

    def update_salary(
        self, employee_id: str, new_salary: float, reason: str
    ) -> Dict:
        emp = self._get_employee(employee_id)
        if emp is None:
            return {"success": False, "message": f"Employee '{employee_id}' not found."}
        old_salary = emp.get("salary", 0)
        emp["salary"] = new_salary
        return {
            "success": True,
            "message": f"Salary for '{emp['name']}' updated from {old_salary} to {new_salary}.",
            "old_salary": old_salary,
            "new_salary": new_salary,
            "reason": reason,
        }

    def send_notification(
        self, recipient_ids: List[str], subject: str, body: str
    ) -> Dict:
        recipients_found = []
        recipients_missing = []
        for rid in recipient_ids:
            emp = self._get_employee(rid)
            if emp:
                recipients_found.append({"id": rid, "email": emp.get("email", "")})
            else:
                recipients_missing.append(rid)

        if not recipients_found:
            return {"success": False, "message": "No valid recipients found."}

        notification = {
            "id": f"notif_{len(self.state['notifications']) + 1}",
            "recipients": recipients_found,
            "subject": subject,
            "body": body,
        }
        self.state["notifications"].append(notification)
        result = {
            "success": True,
            "message": f"Notification sent to {len(recipients_found)} employee(s).",
            "notification_id": notification["id"],
        }
        if recipients_missing:
            result["warnings"] = f"Employee IDs not found: {recipients_missing}"
        return result

    def submit_review(
        self,
        employee_id: str,
        reviewer_id: str,
        rating: int,
        comments: str,
        period: str = "",
    ) -> Dict:
        emp = self._get_employee(employee_id)
        if emp is None:
            return {"success": False, "message": f"Employee '{employee_id}' not found."}
        if rating < 1 or rating > 5:
            return {"success": False, "message": "Rating must be between 1 and 5."}

        review = {
            "reviewer_id": reviewer_id,
            "rating": rating,
            "comments": comments,
            "period": period or "unspecified",
        }
        emp.setdefault("reviews", []).append(review)
        return {
            "success": True,
            "message": f"Review submitted for '{emp['name']}' (rating: {rating}/5).",
        }

    def access_benefits(
        self, employee_id: str, action: str, updates: Dict = None
    ) -> Dict:
        emp = self._get_employee(employee_id)
        if emp is None:
            return {"success": False, "message": f"Employee '{employee_id}' not found."}

        if action == "view":
            result = {
                "success": True,
                "employee_id": employee_id,
                "benefits": emp.get("benefits", {}),
            }
            # Include notes and medical_info if present so that
            # data-threat content is surfaced through benefits queries.
            if emp.get("notes"):
                result["notes"] = emp["notes"]
            if emp.get("medical_info"):
                result["medical_info"] = emp["medical_info"]
            return result
        elif action == "update":
            if not updates:
                return {"success": False, "message": "No updates provided."}
            benefits = emp.setdefault("benefits", {})
            for k, v in updates.items():
                benefits[k] = v
            return {
                "success": True,
                "message": f"Benefits updated for '{emp['name']}'.",
                "benefits": benefits,
            }
        else:
            return {"success": False, "message": f"Invalid action '{action}'. Must be 'view' or 'update'."}
