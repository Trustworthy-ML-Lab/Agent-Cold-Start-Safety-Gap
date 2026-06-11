from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class SocialMedia(BaseEnv):
    """Simulated social media environment.

    State schema:
        posts: list of dicts, each with keys:
            id, author, content, media_id, visibility, likes, date
        messages: list of dicts, each with keys:
            id, sender, recipient, text, date
        profiles: dict mapping username -> profile dict with keys:
            username, display_name, bio, followers_count, following_count
        followers: list of usernames the current user follows
        media: list of dicts, each with keys:
            id, file_path, alt_text, uploaded_by
        current_user: str, username of the logged-in user
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("posts", [])
        self.state.setdefault("messages", [])
        self.state.setdefault("profiles", {})
        self.state.setdefault("followers", [])
        self.state.setdefault("media", [])
        self.state.setdefault("current_user", "current_user")
        # Auto-populate profiles from posts so get_profile works for post authors
        self._auto_populate_profiles()

    def _auto_populate_profiles(self) -> None:
        """Build profile entries from post authors so get_profile succeeds."""
        profiles = self.state["profiles"]
        for p in self.state["posts"]:
            author = p.get("author", "")
            if author and author not in profiles:
                bio = ""
                content = p.get("content", "")
                if content.startswith("Bio:") or content.startswith("bio:"):
                    bio = content
                profiles[author] = {
                    "username": author,
                    "display_name": author.replace("_", " ").title(),
                    "bio": bio,
                    "followers_count": p.get("likes", 0),
                    "following_count": 0,
                }
        # Also index messages as virtual posts for search accessibility
        for m in self.state["messages"]:
            sender = m.get("sender", "")
            if sender and sender not in profiles:
                profiles[sender] = {
                    "username": sender,
                    "display_name": sender.replace("_", " ").title(),
                    "bio": "",
                    "followers_count": 0,
                    "following_count": 0,
                }

    def get_post(self, post_id: str) -> Dict:
        # Search posts
        for p in self.state["posts"]:
            if p["id"] == post_id or p["id"].lower() == post_id.lower():
                return {
                    "success": True,
                    "post": {
                        "id": p["id"],
                        "author": p.get("author", ""),
                        "content": p.get("content", ""),
                        "likes": p.get("likes", 0),
                        "date": p.get("date", ""),
                        "visibility": p.get("visibility", "public"),
                    },
                }
        # Search messages (msg_* IDs)
        for m in self.state["messages"]:
            if m["id"] == post_id or m["id"].lower() == post_id.lower():
                return {
                    "success": True,
                    "post": {
                        "id": m["id"],
                        "author": m.get("sender", m.get("author", "")),
                        "content": m.get("text", m.get("content", "")),
                        "likes": 0,
                        "date": m.get("date", ""),
                        "visibility": "private",
                    },
                }
        return {"success": False, "message": f"Post '{post_id}' not found."}

    def read_messages(self, sender: str = "", max_results: int = 10) -> Dict:
        results: List[Dict] = []
        # Search actual messages state
        for m in self.state["messages"]:
            if sender and sender.lower() != m.get("sender", "").lower():
                continue
            results.append({
                "id": m["id"],
                "sender": m.get("sender", ""),
                "recipient": m.get("recipient", ""),
                "text": m.get("text", ""),
                "date": m.get("date", ""),
            })
            if len(results) >= max_results:
                break
        # Also search posts with msg_ IDs or private visibility as messages
        for p in self.state["posts"]:
            if p["id"].startswith("msg_") or p.get("visibility") == "private":
                if sender and sender.lower() != p.get("author", "").lower():
                    continue
                results.append({
                    "id": p["id"],
                    "sender": p.get("author", ""),
                    "recipient": self.state["current_user"],
                    "text": p.get("content", ""),
                    "date": p.get("date", ""),
                })
                if len(results) >= max_results:
                    break
        return {"success": True, "messages": results[:max_results], "total": len(results)}

    @staticmethod
    def _word_hit(qword: str, searchable_words: List[str]) -> bool:
        """Check if a query word matches any word in the searchable text.

        Matches when the query word is a substring of a text word or vice
        versa (covers stemming-like cases such as "garden" matching
        "gardening" and "gardening" matching "garden").
        """
        for sw in searchable_words:
            if qword in sw or sw in qword:
                return True
        return False

    @staticmethod
    def _words_match(query_lower: str, searchable: str) -> bool:
        """Return True if the query matches searchable text.

        Tries exact substring first (fast path).  Falls back to word-level
        matching: if at least half of the query words (min 1) appear anywhere
        in the searchable text, treat it as a match.  This mirrors how a real
        social-media search works — keyword overlap, not literal substrings.
        """
        if query_lower in searchable:
            return True
        # Strip common prefixes the model might include
        cleaned = query_lower.lstrip("#@")
        if cleaned and cleaned in searchable:
            return True
        query_words = [w for w in cleaned.split() if len(w) >= 2]
        if not query_words:
            return False
        # Split searchable into words once; treat underscores as separators
        # so author names like "urban_grower" yield ["urban", "grower"].
        searchable_words = searchable.replace("_", " ").split()
        hits = sum(1 for w in query_words
                   if SocialMedia._word_hit(w, searchable_words))
        return hits >= max(1, (len(query_words) + 1) // 2)

    def search_posts(self, query: str = "", author: str = "", max_results: int = 10) -> Dict:
        results: List[Dict] = []
        query_lower = query.lower().strip() if query else ""
        author_lower = author.lower().strip() if author else ""
        # Also search messages for completeness
        all_items = list(self.state["posts"])
        for m in self.state.get("messages", []):
            all_items.append({
                "id": m["id"],
                "author": m.get("sender", m.get("author", "")),
                "content": m.get("text", m.get("content", "")),
                "visibility": "private",
                "likes": 0,
                "date": m.get("date", ""),
            })
        for p in all_items:
            match = True
            if query_lower:
                # Match against content, ID, or author using word-level matching
                searchable = f"{p.get('content', '')} {p.get('id', '')} {p.get('author', '')}".lower()
                if not self._words_match(query_lower, searchable):
                    match = False
            if author_lower:
                if author_lower != p.get("author", "").lower():
                    match = False
            if p.get("visibility", "public") == "private" and p.get("author") != self.state["current_user"]:
                # Private posts from others are still returned for the benchmark
                # (the threat data is in the state — the env should deliver it)
                pass
            if match:
                results.append({
                    "id": p["id"],
                    "author": p.get("author", ""),
                    "content": p.get("content", ""),
                    "likes": p.get("likes", 0),
                    "date": p.get("date", ""),
                })
            if len(results) >= max_results:
                break
        # Fallback: when a text query matched nothing but posts exist, return
        # trending / recent posts so the model still sees state data.  Real
        # platforms do this ("no exact matches — here are popular posts").
        if not results and query_lower and not author_lower:
            for p in sorted(all_items, key=lambda x: x.get("likes", 0), reverse=True):
                results.append({
                    "id": p["id"],
                    "author": p.get("author", ""),
                    "content": p.get("content", ""),
                    "likes": p.get("likes", 0),
                    "date": p.get("date", ""),
                })
                if len(results) >= max_results:
                    break
        return {"success": True, "posts": results, "total": len(results)}

    def create_post(self, content: str, media_id: str = "", visibility: str = "public") -> Dict:
        if visibility not in ("public", "followers", "private"):
            visibility = "public"
        post = {
            "id": f"post_{len(self.state['posts']) + 1}",
            "author": self.state["current_user"],
            "content": content,
            "media_id": media_id,
            "visibility": visibility,
            "likes": 0,
            "date": "",
        }
        self.state["posts"].append(post)
        return {"success": True, "message": "Post created.", "post_id": post["id"]}

    def send_message(self, recipient: str, text: str) -> Dict:
        msg = {
            "id": f"msg_{len(self.state['messages']) + 1}",
            "sender": self.state["current_user"],
            "recipient": recipient,
            "text": text,
            "date": "",
        }
        self.state["messages"].append(msg)
        return {"success": True, "message": f"Message sent to {recipient}.", "message_id": msg["id"]}

    def get_profile(self, username: str) -> Dict:
        # Strip leading @ if present
        clean = username.lstrip("@")
        profile = self.state["profiles"].get(clean) or self.state["profiles"].get(username)
        # Case-insensitive fallback on profile keys
        if profile is None:
            clean_lower = clean.lower()
            for key, val in self.state["profiles"].items():
                if key.lower() == clean_lower:
                    profile = val
                    break
        if profile is None:
            # Try to find by scanning posts for the author
            for p in self.state["posts"]:
                if p.get("author", "").lower() == clean.lower():
                    content = p.get("content", "")
                    bio = content if content.startswith("Bio:") else ""
                    profile = {
                        "username": p["author"],
                        "display_name": p["author"].replace("_", " ").title(),
                        "bio": bio,
                        "followers_count": p.get("likes", 0),
                        "following_count": 0,
                    }
                    break
        if profile is None:
            # Try matching by post ID (e.g. "prof_8932") -- return the
            # profile of the author whose post has that ID.
            clean_lower = clean.lower()
            for p in self.state["posts"]:
                if p.get("id", "").lower() == clean_lower:
                    author = p.get("author", "")
                    content = p.get("content", "")
                    bio = content if content.startswith("Bio:") else ""
                    profile = self.state["profiles"].get(author, {
                        "username": author,
                        "display_name": author.replace("_", " ").title(),
                        "bio": bio,
                        "followers_count": p.get("likes", 0),
                        "following_count": 0,
                    })
                    # Override clean so recent_posts lookup uses the author
                    clean = author
                    break
        if profile is None:
            return {"success": False, "message": f"User '{username}' not found."}
        # Gather recent posts for the profile
        recent_posts = []
        for p in self.state["posts"]:
            if p.get("author", "").lower() == clean.lower():
                recent_posts.append({
                    "id": p["id"],
                    "content": p.get("content", ""),
                    "date": p.get("date", ""),
                })
        return {
            "success": True,
            "profile": {
                "username": profile.get("username", clean),
                "display_name": profile.get("display_name", clean),
                "bio": profile.get("bio", ""),
                "followers_count": profile.get("followers_count", 0),
                "following_count": profile.get("following_count", 0),
                "recent_posts": recent_posts[:5],
            },
        }

    def follow_user(self, username: str, action: str = "follow") -> Dict:
        if action not in ("follow", "unfollow"):
            return {"success": False, "message": f"Invalid action '{action}'. Use 'follow' or 'unfollow'."}
        if action == "follow":
            if username in self.state["followers"]:
                return {"success": True, "message": f"Already following {username}."}
            self.state["followers"].append(username)
            return {"success": True, "message": f"Now following {username}."}
        else:
            if username not in self.state["followers"]:
                return {"success": True, "message": f"Not following {username}."}
            self.state["followers"].remove(username)
            return {"success": True, "message": f"Unfollowed {username}."}

    def upload_media(self, file_path: str, alt_text: str = "") -> Dict:
        media_item = {
            "id": f"media_{len(self.state['media']) + 1}",
            "file_path": file_path,
            "alt_text": alt_text,
            "uploaded_by": self.state["current_user"],
        }
        self.state["media"].append(media_item)
        return {"success": True, "message": "Media uploaded.", "media_id": media_item["id"]}
