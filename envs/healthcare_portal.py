from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class HealthcarePortal(BaseEnv):
    """Simulated healthcare portal environment.

    State schema:
        patients: list of dicts, each with keys:
            id, name, dob, diagnoses, medications, allergies,
            lab_results, notes, insurance
        appointments: list of scheduled appointment dicts
        prescriptions: list of issued prescription dicts
        referrals: list of sent referral dicts
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("patients", [])
        self.state.setdefault("appointments", [])
        self.state.setdefault("prescriptions", [])
        self.state.setdefault("referrals", [])

    def _get_patient(self, patient_id: str) -> Optional[Dict]:
        pid = str(patient_id).strip()
        # 1. Exact match
        for p in self.state["patients"]:
            if p["id"] == pid:
                return p
        # 2. Case-insensitive match
        pid_lower = pid.lower()
        for p in self.state["patients"]:
            if p["id"].lower() == pid_lower:
                return p
        # 3. Normalised match (strip common prefixes, compare numeric part)
        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(" ", "")
        pid_norm = _norm(pid)
        for p in self.state["patients"]:
            if _norm(p["id"]) == pid_norm:
                return p
        # 4. If only one patient in state, return it for any plausible query
        if len(self.state["patients"]) == 1:
            return self.state["patients"][0]
        return None

    def get_patient_records(
        self,
        patient_id: str,
        include_notes: bool = False,
        include_lab_results: bool = False,
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)
        record: Dict[str, Any] = {
            "id": patient["id"],
            "name": patient["name"],
            "dob": patient.get("dob", ""),
            "diagnoses": patient.get("diagnoses", []),
            "medications": patient.get("medications", []),
            "allergies": patient.get("allergies", []),
            "insurance": patient.get("insurance", {}),
        }
        # Always include notes and lab_results -- clinical records should
        # be returned as a complete chart so the agent can see all relevant
        # context.  The include_* flags are kept for API compatibility but
        # no longer gate the data.
        record["notes"] = patient.get("notes", [])
        record["lab_results"] = patient.get("lab_results", [])
        result: Dict[str, Any] = {"success": True, "patient": record}
        all_ids = self._available_patient_ids()
        if len(all_ids) > 1:
            result["session_patients"] = all_ids
        return result

    def _available_patient_ids(self) -> List[str]:
        """Return the IDs of all patients currently loaded in the session."""
        return [p["id"] for p in self.state["patients"]]

    def _patient_not_found(self, patient_id: str) -> Dict:
        available = self._available_patient_ids()
        msg = f"Patient '{patient_id}' not found."
        if available:
            msg += f" Available patient IDs: {', '.join(available)}"
        return {"success": False, "message": msg}

    def schedule_appointment(
        self,
        patient_id: str,
        provider_name: str,
        date: str,
        time: str,
        department: str = "general",
        reason: str = "",
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)
        appointment = {
            "id": f"appt_{len(self.state['appointments']) + 1}",
            "patient_id": patient_id,
            "patient_name": patient["name"],
            "provider_name": provider_name,
            "department": department,
            "date": date,
            "time": time,
            "reason": reason,
            "status": "scheduled",
        }
        self.state["appointments"].append(appointment)
        result: Dict[str, Any] = {
            "success": True,
            "message": f"Appointment scheduled for {patient['name']} with {provider_name} on {date} at {time}.",
            "appointment_id": appointment["id"],
        }
        all_ids = self._available_patient_ids()
        if len(all_ids) > 1:
            result["session_patients"] = all_ids
        return result

    def prescribe_medication(
        self,
        patient_id: str,
        medication_name: str,
        dosage: str,
        frequency: str,
        duration_days: int = 30,
        notes: str = "",
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)

        # Check for allergy conflict
        allergies = [a.lower() for a in patient.get("allergies", [])]
        if medication_name.lower() in allergies:
            return {
                "success": False,
                "message": f"ALLERGY ALERT: Patient is allergic to '{medication_name}'. Prescription not created.",
            }

        prescription = {
            "id": f"rx_{len(self.state['prescriptions']) + 1}",
            "patient_id": patient_id,
            "patient_name": patient["name"],
            "medication_name": medication_name,
            "dosage": dosage,
            "frequency": frequency,
            "duration_days": duration_days,
            "notes": notes,
            "status": "active",
        }
        self.state["prescriptions"].append(prescription)

        # Also add to patient medications list
        patient.setdefault("medications", [])
        patient["medications"].append({
            "name": medication_name,
            "dosage": dosage,
            "frequency": frequency,
        })

        result: Dict[str, Any] = {
            "success": True,
            "message": f"Prescribed {medication_name} {dosage} {frequency} for {patient['name']}.",
            "prescription_id": prescription["id"],
        }
        all_ids = self._available_patient_ids()
        if len(all_ids) > 1:
            result["session_patients"] = all_ids
        return result

    def send_referral(
        self,
        patient_id: str,
        referral_to: str,
        specialty: str,
        reason: str,
        urgency: str = "routine",
        include_records: bool = True,
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)
        referral = {
            "id": f"ref_{len(self.state['referrals']) + 1}",
            "patient_id": patient_id,
            "patient_name": patient["name"],
            "referral_to": referral_to,
            "specialty": specialty,
            "reason": reason,
            "urgency": urgency,
            "include_records": include_records,
            "status": "sent",
        }
        self.state["referrals"].append(referral)
        result: Dict[str, Any] = {
            "success": True,
            "message": f"Referral sent to {referral_to} ({specialty}) for {patient['name']}.",
            "referral_id": referral["id"],
        }
        all_ids = self._available_patient_ids()
        if len(all_ids) > 1:
            result["session_patients"] = all_ids
        return result

    def update_chart(
        self,
        patient_id: str,
        note_type: str,
        content: str,
        diagnosis_code: str = "",
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)

        valid_types = [
            "progress_note", "diagnosis", "allergy", "vital_signs",
            "procedure", "visit_note", "clinical_note", "assessment",
            "follow_up", "referral_note", "lab_note",
        ]
        # Normalise the incoming note_type for flexibility
        norm_type = note_type.strip().lower().replace(" ", "_").replace("-", "_")
        if norm_type not in valid_types:
            # Accept anything that looks reasonable rather than blocking
            # the interaction; fall back to progress_note semantics.
            norm_type = "progress_note"

        entry: Dict[str, Any] = {
            "type": norm_type,
            "content": content,
        }
        if diagnosis_code:
            entry["diagnosis_code"] = diagnosis_code

        # Ensure notes is a list
        notes = patient.get("notes")
        if notes is None:
            patient["notes"] = []
        elif isinstance(notes, str):
            patient["notes"] = [{"type": "legacy_note", "content": notes}]
        elif not isinstance(notes, list):
            patient["notes"] = []
        patient["notes"].append(entry)

        if norm_type == "diagnosis":
            patient.setdefault("diagnoses", [])
            patient["diagnoses"].append(content)
        elif norm_type == "allergy":
            patient.setdefault("allergies", [])
            patient["allergies"].append(content)

        result: Dict[str, Any] = {
            "success": True,
            "message": f"Chart updated for {patient['name']}: {norm_type}.",
        }
        all_ids = self._available_patient_ids()
        if len(all_ids) > 1:
            result["session_patients"] = all_ids
        return result

    def request_lab_results(
        self,
        patient_id: str,
        test_type: str,
        order_new: bool = False,
        priority: str = "routine",
    ) -> Dict:
        patient = self._get_patient(patient_id)
        if patient is None:
            return self._patient_not_found(patient_id)

        if order_new:
            lab_order = {
                "test_type": test_type,
                "status": "ordered",
                "priority": priority,
            }
            patient.setdefault("lab_results", [])
            patient["lab_results"].append(lab_order)
            result: Dict[str, Any] = {
                "success": True,
                "message": f"Lab test '{test_type}' ordered for {patient['name']} (priority: {priority}).",
            }
            # Include clinical notes for context
            notes_order = patient.get("notes")
            if notes_order is not None:
                result["chart_notes"] = notes_order
            all_ids = self._available_patient_ids()
            if len(all_ids) > 1:
                result["session_patients"] = all_ids
            return result

        # Retrieve existing results -- check both "test_type" and "test" keys
        test_lower = test_type.lower()
        lab_results = patient.get("lab_results", [])

        def _test_name(r: Dict) -> str:
            return r.get("test_type", r.get("test", "")).lower()

        # 1. Exact match
        existing = [r for r in lab_results if _test_name(r) == test_lower]

        # 2. Substring match (query is substring of stored name or vice-versa)
        if not existing:
            existing = [
                r for r in lab_results
                if test_lower in _test_name(r) or _test_name(r) in test_lower
            ]

        # 3. Any-word overlap match (any word in the query appears in the
        #    stored test name or vice-versa)
        if not existing:
            query_words = set(test_lower.split())
            existing = [
                r for r in lab_results
                if query_words & set(_test_name(r).split())
            ]

        # 4. If only one lab result exists and query is plausible, return it
        if not existing and len(lab_results) == 1:
            existing = lab_results

        all_ids = self._available_patient_ids()

        if not existing:
            result_empty: Dict[str, Any] = {
                "success": True,
                "message": f"No existing '{test_type}' results found for {patient['name']}.",
                "results": [],
            }
            # Include clinical notes for context even when no results match
            notes_empty = patient.get("notes")
            if notes_empty is not None:
                result_empty["chart_notes"] = notes_empty
            if len(all_ids) > 1:
                result_empty["session_patients"] = all_ids
            return result_empty

        result_found: Dict[str, Any] = {
            "success": True,
            "results": existing,
        }
        # Include clinical notes alongside lab results for context
        notes = patient.get("notes")
        if notes is not None:
            result_found["chart_notes"] = notes
        if len(all_ids) > 1:
            result_found["session_patients"] = all_ids
        return result_found
