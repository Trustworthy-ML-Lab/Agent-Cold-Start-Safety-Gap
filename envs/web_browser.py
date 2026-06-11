from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote_plus

from .base_env import BaseEnv


class _OpenForm(dict):
    """A dict that silently accepts any field name.

    ``fill_form`` checks ``field_name not in form`` before setting a value.
    With a regular dict an unknown field triggers an error.  ``_OpenForm``
    overrides ``__contains__`` so every key is considered present, and
    ``__getitem__`` returns ``""`` for missing keys so reads never fail.
    """

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return True

    def __getitem__(self, key: str) -> str:
        return super().get(key, "")


class WebBrowser(BaseEnv):
    """Simulated web browser environment.

    State schema:
        search_results: dict mapping query -> list of result dicts, each with:
            title, url, snippet
        pages: dict mapping url -> content string (page text)
        forms: dict mapping url -> dict of field_name -> current_value
        downloads: list of dicts, each with keys:
            url, save_path, content
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("search_results", {})
        self.state.setdefault("pages", {})
        self.state.setdefault("forms", {})
        self.state.setdefault("downloads", [])
        # _page_meta stores {url: {title, date, ...}} for search enrichment
        self._page_meta: Dict[str, Dict[str, str]] = {}
        pages = self.state["pages"]
        if isinstance(pages, list):
            pages_dict = {}
            for p in pages:
                if isinstance(p, dict) and "url" in p:
                    pages_dict[p["url"]] = p.get("content", "")
                    self._page_meta[p["url"]] = {
                        k: v for k, v in p.items() if k not in ("url", "content")
                    }
            self.state["pages"] = pages_dict
        elif isinstance(pages, dict) and "url" in pages and "content" in pages:
            url = pages["url"]
            self.state["pages"] = {url: pages["content"]}
            self._page_meta[url] = {
                k: v for k, v in pages.items() if k not in ("url", "content")
            }
        # Auto-create forms for every page so fill_form / submit_form can
        # succeed on any URL present in the pages dict.
        self._auto_create_forms()

    # ------------------------------------------------------------------
    # Form auto-creation from page content
    # ------------------------------------------------------------------
    def _auto_create_forms(self) -> None:
        """Create a permissive form for every page URL so that fill_form and
        submit_form always succeed on known pages.  The form accepts any
        field name the model sends (via __missing__-style dict)."""
        for url in list(self.state.get("pages", {}).keys()):
            if url not in self.state["forms"]:
                self.state["forms"][url] = _OpenForm()

    def _get_page_title(self, url: str) -> str:
        """Return the stored title for *url*, or derive one from the URL path."""
        meta = self._page_meta.get(url, {})
        title = meta.get("title", "")
        if not title:
            title = url.split("/")[-1].replace("-", " ").replace("_", " ")
        return title if title.strip() else url

    # ------------------------------------------------------------------
    # URL normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_url(url: str) -> str:
        """Strip trailing slashes, lowcase scheme+host, decode query for comparison."""
        url = url.strip()
        # If URL has no scheme, prepend https:// so urlparse assigns the
        # host correctly instead of putting everything into the path.
        if url and "://" not in url and not url.startswith("//"):
            url = "https://" + url
        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/") or ""
        port_part = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
        # Decode query so that '+' encoding and '%20' encoding match
        query = f"?{unquote_plus(parsed.query)}" if parsed.query else ""
        fragment = f"#{parsed.fragment}" if parsed.fragment else ""
        return f"{scheme}://{host}{port_part}{path}{query}{fragment}"

    def _resolve_page(self, url: str) -> Optional[str]:
        """Try to find page content for a URL with normalisation fallbacks.

        Matching order:
        1. Exact match in pages dict.
        2. Normalised match (trailing slash, scheme, case).
        3. Path-based exact match (handles www vs non-www, http vs https).
        4. Path-prefix match: requested path is a prefix of a stored path
           (same host). Returns the longest stored path that starts with the
           requested path. Handles models visiting parent paths like
           ``/register`` when the page lives at ``/register/2026``.
        """
        pages = self.state.get("pages", {})
        # 1. Exact
        if url in pages:
            return pages[url]
        # 2. Normalised
        norm = self._normalise_url(url)
        for stored_url, content in pages.items():
            if self._normalise_url(stored_url) == norm:
                return content
        # 3. Path-based exact: compare host+path ignoring scheme and www prefix
        try:
            # Ensure scheme is present so urlparse assigns hostname correctly
            _url = url if "://" in url else f"https://{url}"
            req_parsed = urlparse(_url)
            req_host = (req_parsed.hostname or "").lower().removeprefix("www.")
            req_path = req_parsed.path.rstrip("/")
            for stored_url, content in pages.items():
                _stored = stored_url if "://" in stored_url else f"https://{stored_url}"
                sp = urlparse(_stored)
                s_host = (sp.hostname or "").lower().removeprefix("www.")
                s_path = sp.path.rstrip("/")
                if req_host == s_host and req_path == s_path:
                    return content
        except Exception:
            pass
        # 4. Path-prefix: requested path is a non-empty prefix of a stored path
        try:
            _url = url if "://" in url else f"https://{url}"
            req_parsed = urlparse(_url)
            req_host = (req_parsed.hostname or "").lower().removeprefix("www.")
            req_path = req_parsed.path.rstrip("/")
            if req_host and req_path:
                best_content = None
                best_path_len = 0
                for stored_url, content in pages.items():
                    _stored = stored_url if "://" in stored_url else f"https://{stored_url}"
                    sp = urlparse(_stored)
                    s_host = (sp.hostname or "").lower().removeprefix("www.")
                    s_path = sp.path.rstrip("/")
                    if (req_host == s_host
                            and s_path.startswith(req_path)
                            and (len(s_path) == len(req_path)
                                 or s_path[len(req_path)] == "/")):
                        if len(s_path) > best_path_len:
                            best_path_len = len(s_path)
                            best_content = content
                if best_content is not None:
                    return best_content
        except Exception:
            pass
        return None

    def search_web(self, query: str, max_results: int = 10) -> Dict:
        # 1. Exact query match in search_results
        results = list(self.state["search_results"].get(query, []))
        # 2. Word-overlap match on search_results keys
        if not results:
            q_words = [w for w in query.lower().split() if len(w) > 1]
            for key, vals in self.state["search_results"].items():
                if any(w in key.lower() for w in q_words):
                    results.extend(vals)
        # 3. Match against page content, title, and URL using word overlap.
        #    Use a low bar (words > 1 char) so short terms like "IRS" work.
        if not results:
            q_words = [w for w in query.lower().split() if len(w) > 1]
            for url, content in self.state.get("pages", {}).items():
                text = (content or "").lower()
                title = self._get_page_title(url).lower()
                url_lower = unquote_plus(url).lower()
                if any(w in text or w in url_lower or w in title for w in q_words):
                    results.append({
                        "title": self._get_page_title(url),
                        "url": url,
                        "snippet": (content or "")[:200],
                    })
        # 4. Fallback: if still nothing, expose all pages so the model can
        #    discover available content rather than looping on empty results.
        if not results:
            for url, content in self.state.get("pages", {}).items():
                results.append({
                    "title": self._get_page_title(url),
                    "url": url,
                    "snippet": (content or "")[:200],
                })
        results = results[:max_results]
        return {"success": True, "query": query, "results": results, "total": len(results)}

    def visit_url(self, url: str) -> Dict:
        content = self._resolve_page(url)
        if content is None:
            return {"success": False, "message": f"Page '{url}' not found or unreachable."}
        # Ensure a form exists for this URL so subsequent fill_form works
        forms = self.state.get("forms", {})
        if url not in forms:
            forms[url] = _OpenForm()
        return {"success": True, "url": url, "content": content}

    def _resolve_form(self, url: str) -> Optional[Dict]:
        """Find a form by URL with normalisation fallbacks.

        If no form is registered but the URL resolves to a known page,
        create an open form on-the-fly so fill_form / submit_form succeed.
        """
        forms = self.state.get("forms", {})
        if url in forms:
            return forms[url]
        norm = self._normalise_url(url)
        for stored_url, form in forms.items():
            if self._normalise_url(stored_url) == norm:
                return form
        # Path-based match (www vs non-www, scheme)
        try:
            _url = url if "://" in url else f"https://{url}"
            req_parsed = urlparse(_url)
            req_host = (req_parsed.hostname or "").lower().removeprefix("www.")
            req_path = req_parsed.path.rstrip("/")
            for stored_url, form in forms.items():
                _stored = stored_url if "://" in stored_url else f"https://{stored_url}"
                sp = urlparse(_stored)
                s_host = (sp.hostname or "").lower().removeprefix("www.")
                s_path = sp.path.rstrip("/")
                if req_host == s_host and req_path == s_path:
                    return form
        except Exception:
            pass
        # If the URL resolves to a known page, auto-create a form
        if self._resolve_page(url) is not None:
            new_form: Dict = _OpenForm()
            forms[url] = new_form
            return new_form
        return None

    def fill_form(self, url: str, field_name: str, value: str) -> Dict:
        form = self._resolve_form(url)
        if form is None:
            return {"success": False, "message": f"No form found at '{url}'."}
        if field_name not in form:
            return {
                "success": False,
                "message": f"Field '{field_name}' not found in form at '{url}'. Available fields: {', '.join(form.keys())}.",
            }
        form[field_name] = value
        return {"success": True, "message": f"Field '{field_name}' set to '{value}'.", "url": url}

    def download_page(self, url: str, save_path: str = "") -> Dict:
        content = self._resolve_page(url)
        if content is None:
            return {"success": False, "message": f"Page '{url}' not found or unreachable."}
        if not save_path:
            safe_name = url.replace("://", "_").replace("/", "_").replace(".", "_")
            save_path = f"/downloads/{safe_name}.html"
        download = {"url": url, "save_path": save_path, "content": content}
        self.state["downloads"].append(download)
        return {
            "success": True,
            "message": f"Page downloaded and saved to '{save_path}'.",
            "save_path": save_path,
            "size": len(content),
        }

    def submit_form(self, url: str, form_data: dict) -> Dict:
        form = self._resolve_form(url)
        if form is None:
            return {"success": False, "message": f"No form found at '{url}'."}
        for field_name, value in form_data.items():
            if field_name in form:
                form[field_name] = value
        return {
            "success": True,
            "message": f"Form at '{url}' submitted.",
            "submitted_fields": list(form_data.keys()),
        }
