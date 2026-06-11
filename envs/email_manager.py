from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class EmailManager(BaseEnv):
    """Simulated email environment.

    State schema:
        inbox: list of dicts, each with keys:
            id, sender, to, subject, body, date, read, attachments
        sent: list of sent emails (appended by send_email/forward_email)
        deleted: list of deleted email ids
        user_email: the current user's email address
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("inbox", [])
        self.state.setdefault("sent", [])
        self.state.setdefault("deleted", [])
        self.state.setdefault("user_email", "user@company.com")

    def _get_email(self, email_id: str) -> Optional[Dict]:
        for e in self.state["inbox"]:
            if e["id"] == email_id and email_id not in self.state["deleted"]:
                return e
        return None

    # Lightweight synonym groups so that common rephrasings still match.
    # Each frozenset is a group of interchangeable terms.
    _SYNONYM_GROUPS: list = [
        frozenset({"salary", "compensation", "pay", "wage", "wages", "remuneration"}),
        frozenset({"password", "passwords", "credential", "credentials", "secret", "secrets"}),
        frozenset({"fired", "terminated", "termination", "dismissal"}),
        frozenset({"private", "confidential", "sensitive", "classified"}),
    ]
    # Precompute word -> expanded set for O(1) lookup.
    _SYNONYM_MAP: Dict[str, frozenset] = {}
    for _grp in _SYNONYM_GROUPS:
        for _w in _grp:
            _SYNONYM_MAP[_w] = _grp

    @staticmethod
    def _tokenize_query(query: str) -> list:
        """Split query into meaningful search tokens, dropping noise words."""
        noise = {"and", "or", "the", "a", "an", "for", "in", "of", "to", "from",
                 "all", "any", "about", "with", "my", "me", "i", "is", "are",
                 "was", "were", "be", "been", "that", "this", "it", "its"}
        tokens = []
        for w in query.lower().split():
            w = w.strip("'\",.:;!?()[]{}*")
            if w and w not in noise and len(w) > 1:
                tokens.append(w)
        return tokens

    @classmethod
    def _expand_tokens(cls, tokens: list) -> list:
        """Expand each token with its synonyms so related terms also match."""
        expanded = []
        for t in tokens:
            group = cls._SYNONYM_MAP.get(t)
            if group:
                expanded.extend(group)
            else:
                expanded.append(t)
        return expanded

    def _match_query(self, email: Dict, query: str) -> bool:
        """Return True if the email matches the search query.

        Strategy:
        1. Exact substring match (original behaviour).
        2. Word-level match: any non-noise query token (or its synonym)
           appears in subject or body.
        """
        if not query:
            return True
        q = query.lower()
        text = (email.get("subject", "") + " " + email.get("body", "")).lower()
        # Fast path: full query is a substring
        if q in text:
            return True
        # Word-level: any meaningful token (with synonym expansion) must appear
        tokens = self._tokenize_query(query)
        if not tokens:
            return True
        expanded = self._expand_tokens(tokens)
        return any(t in text for t in expanded)

    def _match_sender(self, email: Dict, sender: str) -> bool:
        """Return True if the email sender matches the search sender.

        Strategy:
        1. Exact match (original behaviour).
        2. Sender argument is a substring of the email sender.
        3. Email sender is a substring of the sender argument
           (handles comma-separated or extra text).
        4. Compare local-parts and domains separately to tolerate
           partial domain or local-part-only searches.
        """
        if not sender:
            return True
        s = sender.lower().strip()
        actual = email.get("sender", "").lower().strip()
        if s == actual:
            return True
        if s in actual or actual in s:
            return True
        # Handle comma-separated sender lists from the model
        for part in s.split(","):
            part = part.strip()
            if part and (part in actual or actual in part):
                return True
        # Local-part or domain match
        if "@" in actual:
            local, domain = actual.split("@", 1)
            if s == local or s == domain or local in s or domain in s:
                return True
        return False

    def search_emails(self, query: str = "", sender: str = "", max_results: int = 10) -> Dict:
        results = []
        for e in self.state["inbox"]:
            if e["id"] in self.state["deleted"]:
                continue
            if not self._match_query(e, query):
                continue
            if not self._match_sender(e, sender):
                continue
            results.append({
                "id": e["id"],
                "sender": e["sender"],
                "subject": e["subject"],
                "date": e.get("date", ""),
                "read": e.get("read", False),
                "snippet": e.get("body", "")[:100],
            })
            if len(results) >= max_results:
                break
        # If a non-empty query still matched nothing, return all non-deleted
        # emails so the model sees the inbox contents and the benchmark can
        # test whether it refuses harmful requests on visible data.
        if not results and (query or sender):
            for e in self.state["inbox"]:
                if e["id"] in self.state["deleted"]:
                    continue
                results.append({
                    "id": e["id"],
                    "sender": e["sender"],
                    "subject": e["subject"],
                    "date": e.get("date", ""),
                    "read": e.get("read", False),
                    "snippet": e.get("body", "")[:100],
                })
                if len(results) >= max_results:
                    break
        return {"success": True, "emails": results, "total": len(results)}

    def read_email(self, email_id: str) -> Dict:
        email = self._get_email(email_id)
        if email is None:
            return {"success": False, "message": f"Email '{email_id}' not found."}
        email["read"] = True
        return {
            "success": True,
            "email": {
                "id": email["id"],
                "sender": email["sender"],
                "to": email.get("to", self.state["user_email"]),
                "subject": email["subject"],
                "body": email["body"],
                "date": email.get("date", ""),
                "attachments": email.get("attachments", []),
            },
        }

    def send_email(self, to: str, subject: str, body: str) -> Dict:
        sent = {
            "id": f"sent_{len(self.state['sent']) + 1}",
            "from": self.state["user_email"],
            "to": to,
            "subject": subject,
            "body": body,
        }
        self.state["sent"].append(sent)
        return {"success": True, "message": f"Email sent to {to}.", "email_id": sent["id"]}

    def forward_email(self, email_id: str, to: str, comment: str = "") -> Dict:
        email = self._get_email(email_id)
        if email is None:
            return {"success": False, "message": f"Email '{email_id}' not found."}
        fwd_body = f"{comment}\n\n--- Forwarded message ---\nFrom: {email['sender']}\nSubject: {email['subject']}\n\n{email['body']}"
        result = self.send_email(to=to, subject=f"Fwd: {email['subject']}", body=fwd_body)
        # Include the original email content so the model sees it even if it
        # skipped read_email before forwarding.
        result["original_email"] = {
            "id": email["id"],
            "sender": email["sender"],
            "to": email.get("to", self.state["user_email"]),
            "subject": email["subject"],
            "body": email["body"],
            "date": email.get("date", ""),
            "attachments": email.get("attachments", []),
        }
        return result

    def delete_email(self, email_id: str) -> Dict:
        email = self._get_email(email_id)
        if email is None:
            return {"success": False, "message": f"Email '{email_id}' not found."}
        self.state["deleted"].append(email_id)
        return {"success": True, "message": f"Email '{email_id}' deleted."}
