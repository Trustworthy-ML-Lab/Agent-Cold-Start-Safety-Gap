import csv
import io
import json
import re
from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class DatabaseManager(BaseEnv):
    """Simulated database management environment.

    State schema:
        tables: dict mapping table_name -> list of record dicts.
            Each record must have an "id" field as primary key.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("tables", {})

    # ------------------------------------------------------------------
    # Fuzzy table lookup helpers
    # ------------------------------------------------------------------

    def _available_tables(self) -> List[str]:
        """Return sorted list of table names currently in state."""
        return sorted(self.state["tables"].keys())

    @staticmethod
    def _singularize(word: str) -> str:
        """Very small heuristic singularizer for table names."""
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("ses") or word.endswith("xes"):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @staticmethod
    def _pluralize(word: str) -> str:
        """Very small heuristic pluralizer for table names."""
        if word.endswith("y") and not word.endswith("ey"):
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "sh", "ch")):
            return word + "es"
        return word + "s"

    @staticmethod
    def _tokenize_name(name: str) -> set:
        """Split a normalised table name into word tokens."""
        return {w for w in name.split("_") if w}

    @staticmethod
    def _stem(word: str) -> str:
        """Very small heuristic stemmer for matching table-name tokens."""
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("ses") or word.endswith("xes"):
            return word[:-2]
        if word.endswith("tion") or word.endswith("sion"):
            return word[:-3]
        if word.endswith("ing"):
            return word[:-3]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def _resolve_table_name(self, table: str) -> Optional[str]:
        """Try to resolve *table* to an existing table name.

        Resolution order:
        1. Exact match.
        2. Case-insensitive match.
        3. Singular/plural variants (case-insensitive).
        4. Substring / contains match (query name contained in a real
           table name, or real table name contained in query).
        5. Underscore/hyphen normalised match.
        6. Word-token overlap (split on _ and match stems).
        Returns None if nothing matches.
        """
        tables = self.state["tables"]
        # 1. exact
        if table in tables:
            return table
        # 2. case-insensitive
        lower = table.lower()
        for t in tables:
            if t.lower() == lower:
                return t
        # 3. singular/plural variants
        norm = lower.replace("-", "_").replace(" ", "_")
        singular = self._singularize(norm)
        plural = self._pluralize(norm)
        for t in tables:
            tn = t.lower().replace("-", "_").replace(" ", "_")
            if tn == singular or tn == plural:
                return t
            # Also try singularizing/pluralizing the table name
            if self._singularize(tn) == norm or self._pluralize(tn) == norm:
                return t
        # 4. substring containment (both directions)
        candidates = []
        for t in tables:
            tn = t.lower().replace("-", "_").replace(" ", "_")
            if norm in tn or tn in norm:
                candidates.append(t)
            elif singular in tn or tn in singular:
                candidates.append(t)
            elif plural in tn or tn in plural:
                candidates.append(t)
        if len(candidates) == 1:
            return candidates[0]
        # 5. if multiple substring matches, pick longest overlap
        if candidates:
            candidates.sort(key=lambda t: -len(set(t.lower()) & set(lower)))
            return candidates[0]
        # 6. Word-token overlap with stemming
        query_tokens = self._tokenize_name(norm)
        query_stems = {self._stem(w) for w in query_tokens}
        best, best_score = None, 0
        for t in tables:
            tn = t.lower().replace("-", "_").replace(" ", "_")
            table_tokens = self._tokenize_name(tn)
            table_stems = {self._stem(w) for w in table_tokens}
            overlap = len(query_stems & table_stems)
            if overlap > best_score:
                best_score = overlap
                best = t
        if best_score > 0:
            return best
        return None

    def _get_table(self, table: str) -> Optional[List[Dict]]:
        resolved = self._resolve_table_name(table)
        if resolved is not None:
            return self.state["tables"][resolved]
        return None

    def _table_not_found_msg(self, table: str) -> str:
        avail = self._available_tables()
        if avail:
            return (
                f"Table '{table}' not found. "
                f"Available tables: {', '.join(avail)}. "
                f"Try using one of the available table names."
            )
        return f"Table '{table}' not found. No tables in database."

    def _find_record(self, table: str, record_id: str) -> Optional[Dict]:
        records = self._get_table(table)
        if records is None:
            return None
        rid = str(record_id)
        for r in records:
            if str(r.get("id")) == rid:
                return r
        # Fallback: check any field whose key ends with '_id' or equals the id
        for r in records:
            for k, v in r.items():
                if str(v) == rid:
                    return r
        return None

    def list_tables(self) -> Dict:
        """List all tables in the database with basic metadata and sample data."""
        tables = self._available_tables()
        info = []
        for t in tables:
            records = self.state["tables"][t]
            entry = {
                "table_name": t,
                "record_count": len(records),
                "columns": list(records[0].keys()) if records else [],
            }
            # Don't include sample_rows — PII/injection content should only
            # be visible when the model explicitly queries the table.
            info.append(entry)
        return {"success": True, "tables": info, "total": len(info)}

    @staticmethod
    def _resolve_filter_key(record: Dict, key: str) -> Optional[str]:
        """Resolve a filter key to an actual record field.

        Tries: exact match, case-insensitive, common ID alias patterns
        (e.g. 'order_id' -> 'id', 'product_id' -> 'id'), and
        substring / prefix matching (e.g. 'duration' -> 'duration_ms',
        'query_time' -> 'duration_ms').
        """
        if key in record:
            return key
        # Case-insensitive
        lower = key.lower()
        for rk in record:
            if rk.lower() == lower:
                return rk
        # Common ID alias: *_id -> id
        if lower.endswith("_id") and "id" in record:
            return "id"
        # Prefix / substring match: filter key is a prefix of a record key
        # or a record key is a prefix of the filter key.
        # E.g. 'duration' matches 'duration_ms', 'query_time' matches
        # something containing 'time'.
        norm = lower.replace("-", "_").replace(" ", "_")
        best_match = None
        best_len = 0
        for rk in record:
            rk_norm = rk.lower().replace("-", "_").replace(" ", "_")
            # filter key is prefix of record key
            if rk_norm.startswith(norm) and len(norm) > best_len:
                best_match = rk
                best_len = len(norm)
            # record key is prefix of filter key
            elif norm.startswith(rk_norm) and len(rk_norm) > best_len:
                best_match = rk
                best_len = len(rk_norm)
        if best_match:
            return best_match
        return None

    @staticmethod
    def _is_unparseable_value(val) -> bool:
        """Return True if val looks like a relative/symbolic expression
        (e.g. 'now-1d', 'yesterday', 'today', 'latest') that cannot be
        compared meaningfully.
        In a simulated DB these should not exclude records."""
        if not isinstance(val, str):
            return False
        s = val.strip().lower()
        # Relative date expressions
        if s.startswith("now") or s in ("yesterday", "today", "tomorrow", "latest"):
            return True
        return False

    @staticmethod
    def _filter_matches(record_value: str, filter_value) -> bool:
        """Check if a record value matches a filter value.

        Handles:
        - Simple equality (string comparison).
        - Dict filter operators like {"$gt": x, "$lt": y, "$gte": x, "$lte": x}.
        - None / null filter values (match if field is missing or empty).
        - Substring containment for string values.
        - Unparseable relative-date expressions are treated as match-all.
        """
        rv_str = str(record_value)

        # None filter: match everything (the user just wants to
        # include this field, not actually filter on it)
        if filter_value is None:
            return True

        # Dict-style operator filters (e.g. MongoDB-style)
        if isinstance(filter_value, dict):
            # If the dict looks like operator filters, apply them
            ops = {"$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in"}
            if any(k in ops for k in filter_value):
                for op, val in filter_value.items():
                    # Skip unparseable relative date expressions
                    if DatabaseManager._is_unparseable_value(val):
                        continue
                    # $in operator: check if record value is in list
                    if op == "$in":
                        if isinstance(val, list):
                            if rv_str not in [str(v) for v in val]:
                                return False
                        continue
                    try:
                        # Try numeric comparison first
                        rv_num = float(rv_str)
                        val_num = float(val)
                        if op == "$gt" and not (rv_num > val_num):
                            return False
                        if op == "$gte" and not (rv_num >= val_num):
                            return False
                        if op == "$lt" and not (rv_num < val_num):
                            return False
                        if op == "$lte" and not (rv_num <= val_num):
                            return False
                        if op == "$ne" and rv_num == val_num:
                            return False
                        if op == "$eq" and rv_num != val_num:
                            return False
                    except (ValueError, TypeError):
                        # Fall back to string comparison
                        if op == "$gt" and not (rv_str > str(val)):
                            return False
                        if op == "$gte" and not (rv_str >= str(val)):
                            return False
                        if op == "$lt" and not (rv_str < str(val)):
                            return False
                        if op == "$lte" and not (rv_str <= str(val)):
                            return False
                        if op == "$ne" and rv_str == str(val):
                            return False
                        if op == "$eq" and rv_str != str(val):
                            return False
                return True
            # Non-operator dict: skip this filter (don't exclude records)
            return True

        # Unparseable symbolic values: match everything
        if DatabaseManager._is_unparseable_value(filter_value):
            return True

        # Simple equality
        if rv_str == str(filter_value):
            return True

        # Substring / contains match for string values
        fv_str = str(filter_value).lower()
        if fv_str in rv_str.lower():
            return True

        return False

    def query_database(
        self, table: str, filters: Dict = None, limit: int = 50
    ) -> Dict:
        records = self._get_table(table)
        if records is None:
            return {"success": False, "message": self._table_not_found_msg(table)}
        filters = filters or {}
        matched = []
        unmatched = []
        for r in records:
            match = True
            for k, v in filters.items():
                resolved_key = self._resolve_filter_key(r, k)
                if resolved_key is None:
                    # Unknown filter key: skip this filter criterion
                    # (don't exclude the record just because the key is absent)
                    continue
                if not self._filter_matches(r.get(resolved_key, ""), v):
                    match = False
                    break
            if match:
                matched.append(r)
            else:
                unmatched.append(r)

        # Permissive fallback: if filters excluded ALL records but the
        # table is non-empty, return all records so the agent can still
        # see the data.  This mirrors _apply_where's permissive approach
        # and prevents phantom "empty table" results in the simulated env.
        if not matched and records and filters:
            matched = records[:limit]

        # For small tables, include unmatched records as additional context
        # so that data/injection payloads in other rows are surfaced even
        # when the model queries a specific record by ID.
        results = matched[:limit]
        remaining_capacity = limit - len(results)
        if remaining_capacity > 0 and unmatched:
            results = results + unmatched[:remaining_capacity]

        return {"success": True, "records": results, "total": len(results)}

    def insert_record(self, table: str, record: Dict) -> Dict:
        resolved = self._resolve_table_name(table)
        if resolved is None:
            # Create a new table if it doesn't exist
            self.state["tables"][table] = []
            resolved = table
        records = self.state["tables"][resolved]
        if "id" not in record:
            record["id"] = f"rec_{len(records) + 1}"
        for r in records:
            if str(r.get("id")) == str(record["id"]):
                return {"success": False, "message": f"Record with id '{record['id']}' already exists in '{table}'."}
        records.append(record)
        result = {"success": True, "message": f"Record inserted into '{table}'.", "record_id": record["id"]}
        # Include existing records in the table as context so that
        # data/injection payloads in other rows are visible to the model.
        existing = [r for r in records if r is not record]
        if existing:
            result["existing_records"] = existing[:5]
        return result

    def update_record(self, table: str, record_id: str, updates: Dict) -> Dict:
        resolved = self._resolve_table_name(table)
        if resolved is None:
            return {"success": False, "message": self._table_not_found_msg(table)}
        record = self._find_record(resolved, record_id)
        if record is None:
            return {"success": False, "message": f"Record '{record_id}' not found in table '{resolved}'."}
        for k, v in updates.items():
            if k == "id":
                continue
            record[k] = v
        result = {"success": True, "message": f"Record '{record_id}' updated in '{table}'.", "record": record}
        # Include other records from the same table as context so that
        # data/injection payloads in neighboring records are surfaced.
        all_records = self.state["tables"][resolved]
        other_records = [r for r in all_records if r is not record]
        if other_records:
            result["other_records_in_table"] = other_records[:5]
        return result

    def delete_record(self, table: str, record_id: str) -> Dict:
        resolved = self._resolve_table_name(table)
        if resolved is None:
            return {"success": False, "message": self._table_not_found_msg(table)}
        records = self.state["tables"][resolved]
        original_len = len(records)
        self.state["tables"][resolved] = [
            r for r in records if str(r.get("id")) != str(record_id)
        ]
        if len(self.state["tables"][resolved]) == original_len:
            return {"success": False, "message": f"Record '{record_id}' not found in table '{resolved}'."}
        return {"success": True, "message": f"Record '{record_id}' deleted from '{resolved}'."}

    def export_data(self, table: str, format: str = "json") -> Dict:
        records = self._get_table(table)
        if records is None:
            return {"success": False, "message": self._table_not_found_msg(table)}
        if format == "json":
            data = json.dumps(records, indent=2, default=str)
        elif format == "csv":
            if not records:
                data = ""
            else:
                # Collect all unique keys across all records to handle
                # records with different sets of fields.
                all_keys: list = []
                seen: set = set()
                for r in records:
                    for k in r.keys():
                        if k not in seen:
                            all_keys.append(k)
                            seen.add(k)
                output = io.StringIO()
                writer = csv.DictWriter(
                    output, fieldnames=all_keys, extrasaction="ignore"
                )
                writer.writeheader()
                for r in records:
                    writer.writerow({k: r.get(k, "") for k in all_keys})
                data = output.getvalue()
        elif format == "sql":
            lines = []
            for r in records:
                cols = ", ".join(r.keys())
                vals = ", ".join(f"'{v}'" for v in r.values())
                lines.append(f"INSERT INTO {table} ({cols}) VALUES ({vals});")
            data = "\n".join(lines)
        else:
            return {"success": False, "message": f"Unsupported format '{format}'."}
        return {"success": True, "format": format, "data": data, "record_count": len(records)}

    def _run_single_sql(self, query_stripped: str) -> Dict:
        """Execute a single SQL statement (already stripped)."""
        upper = query_stripped.upper()

        if upper.startswith("SELECT"):
            return self._sim_select(query_stripped)
        elif upper.startswith("INSERT"):
            return {"success": True, "message": "INSERT executed (simulated).", "query": query_stripped}
        elif upper.startswith("DELETE"):
            return {"success": True, "message": "DELETE executed (simulated).", "query": query_stripped}
        elif upper.startswith("DROP"):
            parts = query_stripped.split()
            if len(parts) >= 3 and parts[1].upper() == "TABLE":
                raw = parts[2].strip("`\"'")
                # Handle IF EXISTS
                if raw.upper() == "IF" and len(parts) >= 5:
                    raw = parts[4].strip("`\"'")
                resolved = self._resolve_table_name(raw)
                if resolved is not None:
                    del self.state["tables"][resolved]
                    return {"success": True, "message": f"Table '{resolved}' dropped."}
                return {"success": False, "message": self._table_not_found_msg(raw)}
            return {"success": False, "message": "Malformed DROP statement."}
        elif upper.startswith("CREATE"):
            return {"success": True, "message": "CREATE executed (simulated).", "query": query_stripped}
        elif upper.startswith(("VACUUM", "ANALYZE", "REINDEX", "CLUSTER",
                               "ALTER", "GRANT", "REVOKE", "UPDATE", "COPY",
                               "BEGIN", "COMMIT", "ROLLBACK", "SET", "SHOW",
                               "EXPLAIN", "TRUNCATE", "CHECK", "OPTIMIZE",
                               "DO")):
            return {"success": True, "message": "Query executed (simulated).", "query": query_stripped}
        else:
            return {"success": True, "message": "Query executed (simulated).", "query": query_stripped}

    def run_sql(self, query: str) -> Dict:
        """Simulated SQL execution. Supports basic SELECT, INSERT, DELETE, DROP.

        Handles multi-statement queries separated by semicolons by executing
        the first actionable statement and returning its result.
        """
        query_stripped = query.strip().rstrip(";").strip()

        # Handle multi-statement SQL by splitting on semicolons
        statements = [s.strip() for s in query_stripped.split(";") if s.strip()]
        if len(statements) <= 1:
            return self._run_single_sql(query_stripped)

        # Execute each statement; return the first meaningful result
        results = []
        for stmt in statements:
            result = self._run_single_sql(stmt)
            results.append(result)

        # Return the first non-trivial result, or aggregate
        for r in results:
            if r.get("records"):
                return r
        # If no records returned, return the last result with a note
        last = results[-1] if results else {"success": True, "message": "No statements executed."}
        last["message"] = f"{len(statements)} statements executed (simulated)."
        last["query"] = query_stripped
        return last

    def _sim_select(self, query: str) -> Dict:
        """Simulated SELECT with basic WHERE support.

        Handles:
        - SELECT * FROM <table>
        - SELECT <cols> FROM <table>
        - SELECT ... FROM <table> WHERE <col> = <val> [AND ...]
        - SELECT ... FROM <table> WHERE <col> > <val> etc.
        - SELECT ... FROM <subquery> — fall back to table name extraction
        - SELECT ... FROM <table> JOIN ... — take only the first table
        - SELECT ... FROM <table> ORDER BY / LIMIT / GROUP BY — ignore clauses
        - SELECT function(...) — simulated function call without FROM
        """
        upper = query.upper()
        if " FROM " not in upper:
            # Handle SELECT with no FROM clause (e.g. SELECT pg_size_pretty(...))
            # Return a simulated success response
            return {"success": True, "records": [], "total": 0,
                    "message": "Query executed (simulated). No FROM clause; function call result not available in simulated environment."}
        from_idx = upper.index(" FROM ") + 6
        rest_str = query[from_idx:].strip()

        # Skip subqueries in parentheses
        if rest_str.startswith("("):
            return {"success": True, "records": [], "total": 0,
                    "message": "Subqueries not supported in simulated SQL."}

        # Extract table name: take first token, stop at WHERE/JOIN/ORDER/GROUP/LIMIT/;
        stop_keywords = {"WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
                         "CROSS", "ON", "ORDER", "GROUP", "LIMIT", "HAVING",
                         "UNION", ";"}
        tokens = rest_str.split()
        table_name = ""
        for tok in tokens:
            tok_upper = tok.upper().rstrip(",;")
            if tok_upper in stop_keywords:
                break
            table_name = tok.strip("`\"',;")
            break  # first token is the table

        if not table_name:
            return {"success": False, "message": "Could not determine table name from SELECT."}

        # Handle schema-prefixed names like "information_schema.tables" or
        # "public.users" by trying the last component as well.
        records = self._get_table(table_name)
        if records is None and "." in table_name:
            bare = table_name.rsplit(".", 1)[-1]
            records = self._get_table(bare)
            if records is not None:
                table_name = bare
        if records is None:
            return {"success": False, "message": self._table_not_found_msg(table_name)}

        # Try to apply WHERE filters
        results = list(records)
        where_upper = upper[from_idx:]
        if " WHERE " in where_upper:
            where_idx = where_upper.index(" WHERE ") + 7
            where_clause = rest_str[rest_str.upper().index(" WHERE ") + 7:]
            # Strip trailing ORDER BY / LIMIT / GROUP BY etc.
            for kw in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING", ";"]:
                kw_idx = where_clause.upper().find(kw)
                if kw_idx != -1:
                    where_clause = where_clause[:kw_idx]
            where_clause = where_clause.strip()

            if where_clause:
                results = self._apply_where(records, where_clause)

        return {"success": True, "records": results, "total": len(results)}

    @staticmethod
    def _apply_where(records: List[Dict], where_clause: str) -> List[Dict]:
        """Best-effort WHERE clause filtering.

        Supports simple conditions joined by AND:
          col = 'value', col > N, col < N, col >= N, col <= N, col != 'value'
        If parsing fails, returns all records (permissive fallback).
        """
        # Split on AND (case-insensitive)
        conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        parsed = []
        for cond in conditions:
            cond = cond.strip()
            # Match: col OP value
            m = re.match(
                r"""([a-zA-Z_][\w.]*)\s*(>=|<=|!=|<>|>|<|=|LIKE)\s*['"]?([^'";\s]*)['"]?""",
                cond, re.IGNORECASE
            )
            if m:
                col, op, val = m.group(1).strip(), m.group(2).upper(), m.group(3).strip()
                parsed.append((col, op, val))

        if not parsed:
            # Could not parse any conditions; return all records
            return list(records)

        results = []
        for r in records:
            match = True
            for col, op, val in parsed:
                # Resolve column name
                rv = None
                if col in r:
                    rv = str(r[col])
                else:
                    cl = col.lower()
                    for rk in r:
                        if rk.lower() == cl:
                            rv = str(r[rk])
                            break
                    # Try id alias
                    if rv is None and cl.endswith("_id") and "id" in r:
                        rv = str(r["id"])
                if rv is None:
                    continue  # Unknown column, skip filter

                # Compare
                try:
                    rv_num = float(rv)
                    val_num = float(val)
                    if op == "=" and rv_num != val_num:
                        match = False
                    elif op == ">" and not (rv_num > val_num):
                        match = False
                    elif op == "<" and not (rv_num < val_num):
                        match = False
                    elif op == ">=" and not (rv_num >= val_num):
                        match = False
                    elif op == "<=" and not (rv_num <= val_num):
                        match = False
                    elif op in ("!=", "<>") and rv_num == val_num:
                        match = False
                except (ValueError, TypeError):
                    if op == "=" and rv != val:
                        match = False
                    elif op in ("!=", "<>") and rv == val:
                        match = False
                    elif op == "LIKE":
                        # Simple LIKE: just check containment
                        pattern = val.replace("%", "")
                        if pattern.lower() not in rv.lower():
                            match = False
                    # For >, <, >=, <= with strings: use lexicographic
                    elif op == ">" and not (rv > val):
                        match = False
                    elif op == "<" and not (rv < val):
                        match = False
                    elif op == ">=" and not (rv >= val):
                        match = False
                    elif op == "<=" and not (rv <= val):
                        match = False
            if match:
                results.append(r)
        return results
