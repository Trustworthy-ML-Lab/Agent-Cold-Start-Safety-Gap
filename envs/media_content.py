from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class MediaContent(BaseEnv):
    """Simulated media content generation and publishing environment.

    State schema:
        media_library: list of dicts, each with keys:
            id, type (image/video/audio), prompt, style, status, url, metadata
        publishing_accounts: list of dicts with id, platform, name, connected
        content_queue: list of dicts tracking scheduled publications
        stock_results: dict mapping query strings to lists of stock media results
    """

    # Default publishing accounts so that common template account IDs resolve.
    _DEFAULT_ACCOUNTS = [
        {"id": "acct-instagram-01", "platform": "instagram", "name": "Main Instagram", "connected": True},
        {"id": "acct-instagram-main", "platform": "instagram", "name": "Main Instagram", "connected": True},
        {"id": "acct-facebook-01", "platform": "facebook", "name": "Main Facebook", "connected": True},
        {"id": "acct-youtube-01", "platform": "youtube", "name": "Main YouTube", "connected": True},
        {"id": "acct-youtube-main", "platform": "youtube", "name": "Main YouTube", "connected": True},
        {"id": "acct-linkedin-main", "platform": "linkedin", "name": "Main LinkedIn", "connected": True},
        {"id": "acct-twitter-01", "platform": "twitter", "name": "Main Twitter", "connected": True},
    ]

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("media_library", [])
        if not self.state.get("publishing_accounts"):
            self.state["publishing_accounts"] = [dict(a) for a in self._DEFAULT_ACCOUNTS]
        self.state.setdefault("content_queue", [])
        self.state.setdefault("stock_results", {})

    def _get_media(self, media_id: str) -> Optional[Dict]:
        mid = str(media_id).strip()
        # 1. Exact match
        for m in self.state["media_library"]:
            if m["id"] == mid:
                return m
        # 2. Case-insensitive fallback
        mid_lower = mid.lower()
        for m in self.state["media_library"]:
            if m["id"].lower() == mid_lower:
                return m
        # 3. Normalised match (strip separators)
        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(" ", "")
        mid_norm = _norm(mid)
        for m in self.state["media_library"]:
            if _norm(m["id"]) == mid_norm:
                return m
        # 4. If only one item in library, return it for any plausible query
        if len(self.state["media_library"]) == 1:
            return self.state["media_library"][0]
        return None

    def _media_not_found(self, media_id: str) -> Dict:
        available = [m["id"] for m in self.state.get("media_library", [])]
        msg = f"Media '{media_id}' not found."
        if available:
            msg += f" Available media IDs: {', '.join(available)}"
        return {"success": False, "message": msg}

    def _account_not_found(self, account_id: str) -> Dict:
        available = [a["id"] for a in self.state.get("publishing_accounts", [])]
        msg = f"Publishing account '{account_id}' not found."
        if available:
            msg += f" Available account IDs: {', '.join(available)}"
        return {"success": False, "message": msg}

    def _get_account(self, account_id: str) -> Optional[Dict]:
        aid = str(account_id).strip()
        # Exact match
        for a in self.state["publishing_accounts"]:
            if a["id"] == aid:
                return a
        # Case-insensitive match
        aid_lower = aid.lower()
        for a in self.state["publishing_accounts"]:
            if a["id"].lower() == aid_lower:
                return a
        # Normalised match
        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(" ", "")
        aid_norm = _norm(aid)
        for a in self.state["publishing_accounts"]:
            if _norm(a["id"]) == aid_norm:
                return a
        return None

    def get_media_details(self, media_id: str) -> Dict:
        media = self._get_media(media_id)
        if media is None:
            return self._media_not_found(media_id)
        result = {
            "success": True,
            "media": {
                "id": media["id"],
                "type": media.get("type", "unknown"),
                "prompt": media.get("prompt", ""),
                "style": media.get("style", ""),
                "status": media.get("status", ""),
                "url": media.get("url", ""),
                "metadata": media.get("metadata", {}),
                "transcription": media.get("transcription", ""),
                "edits": media.get("edits", []),
            },
        }
        result.update(self._library_context())
        return result

    def _library_context(self) -> Dict:
        """Return existing media library IDs for context in tool responses."""
        ids = [m["id"] for m in self.state.get("media_library", [])]
        if ids:
            return {"existing_library_items": ids}
        return {}

    def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic",
        resolution: str = "1024x1024",
        negative_prompt: str = "",
    ) -> Dict:
        media_id = f"img_{len(self.state['media_library']) + 1}"
        media_item = {
            "id": media_id,
            "type": "image",
            "prompt": prompt,
            "style": style,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "status": "generated",
            "url": f"https://media.example.com/{media_id}.png",
            "metadata": {"resolution": resolution, "style": style},
        }
        self.state["media_library"].append(media_item)
        result = {
            "success": True,
            "message": f"Image generated from prompt.",
            "media_id": media_id,
            "url": media_item["url"],
        }
        result.update(self._library_context())
        return result

    def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 10,
        source_media_id: str = "",
        style: str = "cinematic",
    ) -> Dict:
        if source_media_id:
            source = self._get_media(source_media_id)
            if source is None:
                return self._media_not_found(source_media_id)

        media_id = f"vid_{len(self.state['media_library']) + 1}"
        media_item = {
            "id": media_id,
            "type": "video",
            "prompt": prompt,
            "style": style,
            "duration_seconds": duration_seconds,
            "source_media_id": source_media_id,
            "status": "generated",
            "url": f"https://media.example.com/{media_id}.mp4",
            "metadata": {"duration": duration_seconds, "style": style},
        }
        self.state["media_library"].append(media_item)
        result = {
            "success": True,
            "message": f"Video generated ({duration_seconds}s).",
            "media_id": media_id,
            "url": media_item["url"],
        }
        result.update(self._library_context())
        return result

    def edit_media(
        self,
        media_id: str,
        operation: str,
        parameters: Optional[Dict] = None,
    ) -> Dict:
        media = self._get_media(media_id)
        if media is None:
            return self._media_not_found(media_id)

        valid_ops = ["crop", "resize", "filter", "overlay_text", "merge", "trim", "add_audio", "blur", "brightness", "contrast"]
        if operation not in valid_ops:
            return {"success": False, "message": f"Invalid operation '{operation}'. Must be one of {valid_ops}."}

        params = parameters or {}
        media.setdefault("edits", [])
        media["edits"].append({"operation": operation, "parameters": params})
        media["status"] = "edited"
        return {
            "success": True,
            "message": f"Applied '{operation}' to media '{media_id}'.",
            "media_id": media_id,
        }

    def publish_content(
        self,
        media_id: str,
        account_id: str,
        caption: str = "",
        tags: Optional[List[str]] = None,
        schedule_time: str = "",
    ) -> Dict:
        media = self._get_media(media_id)
        if media is None:
            return self._media_not_found(media_id)
        account = self._get_account(account_id)
        if account is None:
            return self._account_not_found(account_id)
        if not account.get("connected", True):
            return {"success": False, "message": f"Account '{account_id}' is not connected."}

        pub_entry = {
            "id": f"pub_{len(self.state['content_queue']) + 1}",
            "media_id": media_id,
            "account_id": account_id,
            "platform": account.get("platform", "unknown"),
            "caption": caption,
            "tags": tags or [],
            "schedule_time": schedule_time,
            "status": "scheduled" if schedule_time else "published",
        }
        self.state["content_queue"].append(pub_entry)
        status_msg = f"scheduled for {schedule_time}" if schedule_time else "published immediately"
        return {
            "success": True,
            "message": f"Content {status_msg} to {account.get('platform', 'unknown')}.",
            "publication_id": pub_entry["id"],
        }

    def search_stock_media(
        self,
        query: str,
        media_type: str = "image",
        license_type: str = "commercial",
        max_results: int = 10,
    ) -> Dict:
        stock = self.state.get("stock_results", {})
        key = query.lower()

        # --- Phase 1: collect from stock_results -------------------------
        stock_hits: List[Dict] = []
        exact_stock = stock.get(key, [])
        if exact_stock:
            stock_hits.extend(exact_stock)
        else:
            for k, v in stock.items():
                if any(w in k.lower() for w in key.split()):
                    stock_hits.extend(v)

        # --- Phase 2: collect from media_library -------------------------
        # Always search media_library so that items seeded in state are
        # discoverable even when stock_results also matched.
        id_match_ids: set = set()
        library_id_hits: List[Dict] = []
        library_kw_hits: List[Dict] = []
        seen_ids: set = set()

        for m in self.state.get("media_library", []):
            m_id_lower = m.get("id", "").lower()
            if m_id_lower == key or key in m_id_lower:
                if m["id"] not in seen_ids:
                    library_id_hits.append(m)
                    id_match_ids.add(m["id"])
                    seen_ids.add(m["id"])

        for m in self.state.get("media_library", []):
            if m["id"] in seen_ids:
                continue
            desc = f"{m.get('prompt', '')} {m.get('type', '')} {m.get('id', '')}".lower()
            if any(w in desc for w in key.split()):
                library_kw_hits.append(m)
                seen_ids.add(m["id"])

        # --- Phase 3: merge and filter by type ---------------------------
        # ID matches from media_library always bypass the type filter.
        # Everything else is filtered by media_type.
        filtered: List[Dict] = []
        included_ids: set = set()

        # Always include library ID matches (no type filter)
        for item in library_id_hits:
            filtered.append(item)
            included_ids.add(item["id"])

        # Type-filter stock hits
        for item in stock_hits:
            iid = item.get("id", id(item))
            if iid in included_ids:
                continue
            if item.get("type", "image") == media_type:
                filtered.append(item)
                included_ids.add(iid)

        # Type-filter library keyword hits
        for item in library_kw_hits:
            if item["id"] in included_ids:
                continue
            if item.get("type", "image") == media_type:
                filtered.append(item)
                included_ids.add(item["id"])

        return {
            "success": True,
            "query": query,
            "media_type": media_type,
            "results": filtered[:max_results],
            "total": len(filtered),
        }

    def transcribe_audio(
        self,
        media_id: str,
        language: str = "en",
        include_timestamps: bool = False,
    ) -> Dict:
        media = self._get_media(media_id)
        if media is None:
            return self._media_not_found(media_id)
        if media.get("type") not in ("video", "audio"):
            return {"success": False, "message": f"Media '{media_id}' is not an audio or video file."}

        # Return pre-configured transcription if available.
        # Fall back to the prompt field (which may contain content intended
        # to be surfaced via transcription), then to a placeholder.
        transcription = (
            media.get("transcription")
            or media.get("prompt")
            or f"[Transcription of {media_id} in {language}]"
        )
        result: Dict[str, Any] = {
            "success": True,
            "media_id": media_id,
            "language": language,
            "text": transcription,
        }
        if include_timestamps:
            result["segments"] = media.get("transcription_segments", [])
        result.update(self._library_context())
        return result
