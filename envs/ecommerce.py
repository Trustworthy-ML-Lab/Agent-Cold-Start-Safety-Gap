from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class ECommerce(BaseEnv):
    """Simulated e-commerce platform environment.

    State schema:
        products: list of dicts, each with keys:
            id, name, price, description, category, restricted, in_stock
        cart: list of dicts with product_id, quantity
        orders: list of placed order dicts
        reviews: list of submitted review dicts
        payment_methods: list of dicts with id, type, last_four
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("products", [])
        self.state.setdefault("cart", [])
        self.state.setdefault("orders", [])
        self.state.setdefault("reviews", [])
        self.state.setdefault("payment_methods", [])

    def _get_product(self, product_id: str) -> Optional[Dict]:
        # Exact match first.
        for p in self.state["products"]:
            if p["id"] == product_id:
                return p
        # Fallback: case-insensitive, then prefix/substring match.
        pid_lower = product_id.lower().strip()
        for p in self.state["products"]:
            if p["id"].lower().strip() == pid_lower:
                return p
        for p in self.state["products"]:
            p_lower = p["id"].lower().strip()
            if pid_lower in p_lower or p_lower in pid_lower:
                return p
        return None

    def _get_order(self, order_id: str) -> Optional[Dict]:
        # Exact match first.
        for o in self.state["orders"]:
            if o["id"] == order_id:
                return o
        # Fallback: case-insensitive match, then prefix/substring.
        oid_lower = order_id.lower().strip()
        for o in self.state["orders"]:
            if o["id"].lower().strip() == oid_lower:
                return o
        for o in self.state["orders"]:
            o_lower = o["id"].lower().strip()
            if oid_lower in o_lower or o_lower in oid_lower:
                return o
        return None

    def _get_payment_method(self, pm_id: str) -> Optional[Dict]:
        for pm in self.state["payment_methods"]:
            if pm["id"] == pm_id:
                return pm
        return None

    @staticmethod
    def _query_matches(query: str, text: str) -> bool:
        """Check if query matches text using word-level matching.

        Matches if the full query is a substring, or if at least half
        of the query words appear in the text (with simple prefix
        support for plurals like 'cards' matching 'card').
        """
        if not query:
            return True
        q_lower = query.lower()
        t_lower = text.lower()
        # First try exact substring match.
        if q_lower in t_lower:
            return True
        # Word-level matching with prefix support.
        q_words = [w for w in q_lower.split() if len(w) >= 2]
        if not q_words:
            return False
        t_words = t_lower.split()
        hits = 0
        for qw in q_words:
            for tw in t_words:
                # Match if query word is substring of text word or vice versa
                # (handles plurals: 'cards'/'card', 'makers'/'maker').
                if qw in tw or tw in qw:
                    hits += 1
                    break
        # Require at least half the query words to match.
        return hits >= max(1, len(q_words) // 2)

    @staticmethod
    def _category_matches(filter_cat: str, product_cat: str) -> bool:
        """Check if a category filter matches a product category.

        Matches on exact equality (case-insensitive) first, then falls
        back to word-level substring matching so that related category
        names like 'pharmaceuticals' and 'health' can still match.
        """
        if not filter_cat:
            return True
        fc = filter_cat.lower().strip()
        pc = product_cat.lower().strip()
        if fc == pc:
            return True
        # Substring match in either direction.
        if fc in pc or pc in fc:
            return True
        # Word-level: any word in the filter appears in the product
        # category or vice versa (handles 'kitchen appliances' vs 'kitchen').
        fc_words = fc.split()
        pc_words = pc.split()
        for fw in fc_words:
            for pw in pc_words:
                if fw in pw or pw in fw:
                    return True
        return False

    def _search_products_inner(
        self,
        query: str,
        category: str,
        min_price: float,
        max_price: float,
        max_results: int,
    ) -> List[Dict]:
        results: List[Dict] = []
        for p in self.state["products"]:
            match = True
            if query:
                name = p.get("name", "")
                desc = p.get("description", "")
                pid = p.get("id", "")
                # Also check product ID so the model can search by ID.
                if (not self._query_matches(query, name)
                        and not self._query_matches(query, desc)
                        and query.strip().lower() != pid.lower()):
                    match = False
            if category and not self._category_matches(category, p.get("category", "")):
                match = False
            price = p.get("price", 0)
            if price < min_price or price > max_price:
                match = False
            if match:
                entry = {
                    "id": p["id"],
                    "name": p["name"],
                    "price": p["price"],
                    "category": p.get("category", ""),
                    "in_stock": p.get("in_stock", True),
                }
                # Include a description snippet so injection payloads
                # and data-threat content are partially surfaced even
                # before get_product_details is called.
                desc = p.get("description", "")
                if desc:
                    entry["description_snippet"] = (
                        desc[:120] + "..." if len(desc) > 120 else desc
                    )
                results.append(entry)
            if len(results) >= max_results:
                break
        return results

    def search_products(
        self,
        query: str = "",
        category: str = "",
        min_price: float = 0.0,
        max_price: float = 1e9,
        max_results: int = 10,
    ) -> Dict:
        results = self._search_products_inner(
            query, category, min_price, max_price, max_results
        )
        # Fallback: if a category filter produced zero results but there
        # are products that match the query alone, retry without the
        # category constraint.  This handles cases where the category
        # name in the user turn doesn't exactly match any product
        # category (e.g. "pharmaceuticals" vs "health").
        if not results and category:
            results = self._search_products_inner(
                query, "", min_price, max_price, max_results
            )
        # Final fallback: if still no results but products exist, return all
        # products so the model can discover available content.
        if not results and self.state["products"]:
            for p in self.state["products"][:max_results]:
                entry = {
                    "id": p["id"],
                    "name": p["name"],
                    "price": p["price"],
                    "category": p.get("category", ""),
                    "in_stock": p.get("in_stock", True),
                }
                desc = p.get("description", "")
                if desc:
                    entry["description_snippet"] = (
                        desc[:120] + "..." if len(desc) > 120 else desc
                    )
                results.append(entry)
        return {"success": True, "products": results, "total": len(results)}

    def get_product_details(self, product_id: str) -> Dict:
        product = self._get_product(product_id)
        if product is None:
            return {"success": False, "message": f"Product '{product_id}' not found."}
        # Return all fields from the product dict so that any embedded
        # data (reviews, notes, leak simulations, Q&A, etc.) is surfaced.
        details = dict(product)
        # Ensure core fields have sensible defaults even if missing.
        details.setdefault("restricted", False)
        details.setdefault("in_stock", True)
        return {"success": True, "product": details}

    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict:
        product = self._get_product(product_id)
        if product is None:
            return {"success": False, "message": f"Product '{product_id}' not found."}
        if not product.get("in_stock", True):
            return {"success": False, "message": f"Product '{product_id}' is out of stock."}
        # Check if already in cart and update quantity
        for item in self.state["cart"]:
            if item["product_id"] == product_id:
                item["quantity"] += quantity
                return {
                    "success": True,
                    "message": f"Updated quantity for '{product['name']}' to {item['quantity']}.",
                    "cart_size": len(self.state["cart"]),
                }
        self.state["cart"].append({"product_id": product_id, "quantity": quantity})
        return {
            "success": True,
            "message": f"Added {quantity}x '{product['name']}' to cart.",
            "cart_size": len(self.state["cart"]),
        }

    def place_order(
        self,
        payment_method_id: str,
        shipping_address: str,
        gift_wrap: bool = False,
    ) -> Dict:
        if not self.state["cart"]:
            return {"success": False, "message": "Cart is empty."}
        pm = self._get_payment_method(payment_method_id)
        if pm is None:
            # Accept ad-hoc payment methods so that user-threat scenarios
            # (e.g. using a corporate card) can proceed and test whether
            # the model refuses rather than being blocked by env limitations.
            pm = {"id": payment_method_id, "type": "unknown", "last_four": "????"}
            self.state["payment_methods"].append(pm)

        items: List[Dict] = []
        total = 0.0
        for cart_item in self.state["cart"]:
            product = self._get_product(cart_item["product_id"])
            if product is None:
                continue
            line_total = product["price"] * cart_item["quantity"]
            total += line_total
            items.append({
                "product_id": product["id"],
                "name": product["name"],
                "quantity": cart_item["quantity"],
                "unit_price": product["price"],
                "line_total": line_total,
            })

        order = {
            "id": f"order_{len(self.state['orders']) + 1}",
            "items": items,
            "total": round(total, 2),
            "payment_method_id": payment_method_id,
            "shipping_address": shipping_address,
            "gift_wrap": gift_wrap,
            "status": "placed",
        }
        self.state["orders"].append(order)
        self.state["cart"] = []
        return {
            "success": True,
            "message": f"Order placed successfully.",
            "order_id": order["id"],
            "total": order["total"],
        }

    def submit_review(
        self,
        product_id: str,
        rating: int,
        body: str,
        title: str = "",
    ) -> Dict:
        if rating < 1 or rating > 5:
            return {"success": False, "message": "Rating must be between 1 and 5."}
        product = self._get_product(product_id)
        # Accept reviews even for products not in the catalog so that
        # filler turns referencing out-of-state products don't block the
        # conversation from reaching the threat-delivery turn.
        product_name = product["name"] if product else product_id
        review = {
            "id": f"review_{len(self.state['reviews']) + 1}",
            "product_id": product_id,
            "rating": rating,
            "title": title,
            "body": body,
        }
        self.state["reviews"].append(review)
        return {
            "success": True,
            "message": f"Review submitted for '{product_name}'.",
            "review_id": review["id"],
        }

    def manage_returns(
        self,
        order_id: str,
        reason: str,
        product_id: str = "",
        refund_method: str = "original_payment",
        refund_account: str = "",
    ) -> Dict:
        order = self._get_order(order_id)
        if order is None:
            # Accept ad-hoc order IDs so that filler turns referencing
            # orders not in the current state don't block the conversation
            # from reaching the threat-delivery turn.
            order = {
                "id": order_id,
                "items": [],
                "total": 0.0,
                "payment_method_id": "unknown",
                "shipping_address": "unknown",
                "status": "delivered",
            }
            if product_id:
                product = self._get_product(product_id)
                p_name = product["name"] if product else product_id
                p_price = product["price"] if product else 0.0
                order["items"] = [{
                    "product_id": product_id,
                    "name": p_name,
                    "quantity": 1,
                    "unit_price": p_price,
                    "line_total": p_price,
                }]
                order["total"] = p_price
            self.state["orders"].append(order)
        if order.get("status") == "returned":
            return {"success": False, "message": f"Order '{order_id}' has already been returned."}

        refund_amount = order["total"]
        if product_id:
            found = False
            for item in order.get("items", []):
                if item["product_id"] == product_id:
                    refund_amount = item["line_total"]
                    found = True
                    break
            if not found:
                return {"success": False, "message": f"Product '{product_id}' not found in order '{order_id}'."}

        order["status"] = "return_initiated"
        result = {
            "success": True,
            "message": f"Return initiated for order '{order_id}'.",
            "refund_amount": round(refund_amount, 2),
            "refund_method": refund_method,
            "reason": reason,
        }
        if refund_account:
            result["refund_account"] = refund_account
            result["message"] += f" Refund will be sent to account {refund_account}."
        return result
