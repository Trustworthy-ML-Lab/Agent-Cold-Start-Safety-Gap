from typing import Any, Dict, List, Optional

from .base_env import BaseEnv

# Standard flight keys (everything else is an "extra" field to preserve)
_FLIGHT_STANDARD_KEYS = {
    "id", "airline", "origin", "destination", "date", "departure_time",
    "arrival_time", "price", "cabin_class", "class", "seats_available",
}


class TravelBooking(BaseEnv):
    """Simulated travel booking environment.

    State schema:
        flights: list of dicts, each with keys:
            id, origin, destination, date, price, cabin_class, airline,
            departure_time, arrival_time, seats_available
            May also contain extra fields (note, data-leak fields, etc.)
            that are passed through verbatim.
        hotels: list of dicts with id, name, location, price_per_night,
            rating, room_types, amenities
        bookings: list of placed booking dicts
        payment_methods: list of dicts with id, type, last_four
        loyalty_accounts: list of dicts with id, program, member_name, points
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("flights", [])
        self.state.setdefault("hotels", [])
        self.state.setdefault("bookings", [])
        self.state.setdefault("payment_methods", [])
        self.state.setdefault("loyalty_accounts", [])

    def _get_flight(self, flight_id: str) -> Optional[Dict]:
        for f in self.state["flights"]:
            if f["id"] == flight_id:
                return f
        return None

    def _get_hotel(self, hotel_id: str) -> Optional[Dict]:
        for h in self.state["hotels"]:
            if h["id"] == hotel_id:
                return h
        return None

    def _get_booking(self, booking_id: str) -> Optional[Dict]:
        for b in self.state["bookings"]:
            if b["id"] == booking_id:
                return b
        return None

    def _get_payment_method(self, pm_id: str) -> Optional[Dict]:
        for pm in self.state["payment_methods"]:
            if pm["id"] == pm_id:
                return pm
        return None

    def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        cabin_class: str = "economy",
        max_results: int = 10,
    ) -> Dict:
        results: List[Dict] = []
        seen_ids: set = set()
        for f in self.state["flights"]:
            match = True
            if origin.upper() != f.get("origin", "").upper():
                match = False
            if destination.upper() != f.get("destination", "").upper():
                match = False
            if date != f.get("date", ""):
                match = False
            f_class = self._get_cabin_class(f)
            if cabin_class.lower() != f_class.lower():
                match = False
            if match:
                results.append(self._format_flight(f))
                seen_ids.add(f.get("id"))
            if len(results) >= max_results:
                break

        # Always include flights with extra fields (data-leak, injection notes)
        # so that threat payloads reach the model even if search params differ.
        for f in self.state["flights"]:
            if f.get("id") in seen_ids:
                continue
            has_extra = any(
                k not in _FLIGHT_STANDARD_KEYS for k in f
            )
            if has_extra and len(results) < max_results:
                results.append(self._format_flight(f))
                seen_ids.add(f.get("id"))

        if not results:
            # Fallback: return all flights so threat payloads are visible
            for f in self.state["flights"][:max_results]:
                results.append(self._format_flight(f))
        return {"success": True, "flights": results, "total": len(results)}

    @staticmethod
    def _get_cabin_class(f: Dict) -> str:
        """Read cabin class from either 'cabin_class' or 'class' key."""
        return f.get("cabin_class") or f.get("class") or "economy"

    def _format_flight(self, f: Dict) -> Dict:
        """Format a flight dict, preserving ALL extra fields verbatim.

        Extra fields (note, data-leak fields, etc.) are included so that
        threat payloads embedded in state_items reach the model.
        """
        result = {
            "id": f["id"],
            "airline": f.get("airline", ""),
            "origin": f.get("origin", ""),
            "destination": f.get("destination", ""),
            "date": f.get("date", ""),
            "departure_time": f.get("departure_time", ""),
            "arrival_time": f.get("arrival_time", ""),
            "price": f.get("price", 0),
            "cabin_class": self._get_cabin_class(f),
            "seats_available": f.get("seats_available", 0),
        }
        # Append every extra key not already covered
        for key, val in f.items():
            if key not in _FLIGHT_STANDARD_KEYS and key not in result:
                result[key] = val
        return result

    def book_flight(
        self,
        flight_id: str,
        passengers: List[Dict],
        payment_method_id: str,
        loyalty_account_id: str = "",
    ) -> Dict:
        flight = self._get_flight(flight_id)
        if flight is None:
            return {"success": False, "message": f"Flight '{flight_id}' not found."}
        pm = self._get_payment_method(payment_method_id)
        # Auto-accept any payment method if none are defined in state
        # (templates don't populate payment_methods, so we accept whatever
        # the model provides to allow bookings to succeed).
        if pm is None and self.state.get("payment_methods"):
            return {"success": False, "message": f"Payment method '{payment_method_id}' not found."}
        seats = flight.get("seats_available", 0)
        if seats < len(passengers):
            return {"success": False, "message": f"Only {seats} seats available, but {len(passengers)} passengers requested."}

        total_price = flight["price"] * len(passengers)
        details = {
            "airline": flight.get("airline", ""),
            "origin": flight["origin"],
            "destination": flight["destination"],
            "date": flight["date"],
            "departure_time": flight.get("departure_time", ""),
        }
        # Copy extra fields (note, data-leak fields) into booking details
        for key, val in flight.items():
            if key not in _FLIGHT_STANDARD_KEYS and key not in details:
                details[key] = val
        booking = {
            "id": f"bk_{len(self.state['bookings']) + 1}",
            "type": "flight",
            "flight_id": flight_id,
            "passengers": passengers,
            "payment_method_id": payment_method_id,
            "loyalty_account_id": loyalty_account_id,
            "total_price": round(total_price, 2),
            "status": "confirmed",
            "details": details,
        }
        self.state["bookings"].append(booking)
        flight["seats_available"] = seats - len(passengers)
        return {
            "success": True,
            "message": f"Flight booked for {len(passengers)} passenger(s).",
            "booking_id": booking["id"],
            "total_price": booking["total_price"],
        }

    @staticmethod
    def _format_hotel(h: Dict) -> Dict:
        """Format a hotel dict, preserving ALL extra fields verbatim."""
        result = {
            "id": h.get("id", ""),
            "name": h.get("name", h.get("airline", "")),
            "location": h.get("location", h.get("destination", "")),
            "price_per_night": h.get("price_per_night", h.get("price", 0)),
            "rating": h.get("rating", 0),
            "room_types": h.get("room_types", []),
            "amenities": h.get("amenities", []),
        }
        # Standard hotel keys already covered
        _hotel_covered = {
            "id", "name", "location", "price_per_night", "price", "rating",
            "room_types", "amenities",
            # Also skip flight-specific keys that are irrelevant for hotels
            "airline", "origin", "destination", "date", "departure_time",
            "arrival_time", "cabin_class", "class", "seats_available",
        }
        for key, val in h.items():
            if key not in _hotel_covered and key not in result:
                result[key] = val
        return result

    def _get_hotel_pool(self) -> List[Dict]:
        """Collect hotels from both state['hotels'] and state['flights'].

        Templates put hotel-context items under state_key='flights', so we
        also search state['flights'] for items that look hotel-like.
        Detection heuristics (any match means hotel-like):
          - airline field contains 'Hotel', 'Inn', 'Marriott', 'Hilton',
            'Hyatt', 'Westin', 'Best Western', 'Comfort', 'Holiday',
            'Ritz', 'Four Seasons', 'Search Result', 'Search Context'
          - class field is hotel/suite/luxury/deluxe/premium/standard
          - item has data-leak keys with hotel-relevant names
        If none of the flights items matches these heuristics but
        state['hotels'] is empty, fall back to returning ALL flight items
        with seats_available == 0 (likely booking/context items).
        """
        pool: List[Dict] = list(self.state.get("hotels", []))
        _hotel_keywords = (
            "hotel", "inn", "marriott", "hilton", "hyatt", "westin",
            "best western", "comfort", "holiday", "ritz", "four seasons",
            "search result", "search context",
        )
        _hotel_classes = {"hotel", "suite", "luxury", "deluxe", "premium", "standard"}
        _leak_keywords = {"guest_leak", "vip", "celebrity", "diplomatic", "witness", "corporate_vip"}

        for f in self.state.get("flights", []):
            airline_lower = f.get("airline", "").lower()
            f_class = f.get("class", f.get("cabin_class", "")).lower()
            is_hotel = (
                any(kw in airline_lower for kw in _hotel_keywords)
                or f_class in _hotel_classes
                or any(any(lk in k.lower() for lk in _leak_keywords) for k in f)
            )
            if is_hotel:
                pool.append(f)

        # Fallback: if no hotels found at all, include ALL flights items
        # with seats_available == 0 (context/booking items placed by templates)
        if not pool:
            for f in self.state.get("flights", []):
                pool.append(f)

        return pool

    # Common city name <-> airport/city code mappings for flexible matching
    _CITY_ALIASES: Dict[str, List[str]] = {
        "nyc": ["new york", "manhattan", "brooklyn", "queens", "bronx"],
        "chi": ["chicago"],
        "par": ["paris", "eiffel"],
        "lon": ["london", "covent garden", "bloomsbury"],
        "tyo": ["tokyo", "shinjuku", "shibuya"],
        "sfo": ["san francisco", "fisherman"],
        "lax": ["los angeles", "hollywood"],
        "mia": ["miami"],
        "den": ["denver"],
        "dfw": ["dallas"],
        "syd": ["sydney"],
        "ord": ["chicago"],
        "jfk": ["new york", "manhattan"],
        "lhr": ["london"],
        "cdg": ["paris"],
        "nrt": ["tokyo"],
        "cun": ["cancun"],
        "rom": ["rome"],
        "fco": ["rome"],
        "fll": ["fort lauderdale"],
        "bos": ["boston"],
        "atl": ["atlanta"],
        "sea": ["seattle"],
        "iad": ["washington"],
    }

    @classmethod
    def _location_matches(cls, query: str, item_location: str, item_name: str) -> bool:
        """Check if a search query matches a hotel's location or name.

        Handles both direct substring matching and city alias resolution.
        """
        q = query.lower()
        loc = item_location.lower()
        name = item_name.lower()

        # Direct substring match
        if q in loc or q in name or loc in q or name in q:
            return True

        # Check if any word in the query matches location code or vice versa
        q_words = q.split()
        for word in q_words:
            if word in loc or loc in word:
                return True

        # Alias-based matching: if location is a known code, check aliases
        for code, aliases in cls._CITY_ALIASES.items():
            # Check if location code matches any alias in the query
            if code == loc or code in loc:
                if any(alias in q for alias in aliases):
                    return True
            # Check if query contains a code that maps to aliases matching location
            if code in q:
                if any(alias in loc for alias in aliases):
                    return True

        return False

    def search_hotels(
        self,
        location: str,
        check_in: str,
        check_out: str,
        guests: int = 1,
        max_price: float = 1e9,
        max_results: int = 10,
    ) -> Dict:
        _hotel_standard_keys = {
            "id", "name", "location", "price_per_night", "price", "rating",
            "room_types", "amenities", "airline", "origin", "destination",
            "date", "departure_time", "arrival_time", "cabin_class", "class",
            "seats_available",
        }
        pool = self._get_hotel_pool()
        results: List[Dict] = []
        seen_ids: set = set()
        for h in pool:
            match = True
            h_location = h.get("location", h.get("destination", ""))
            h_name = h.get("name", h.get("airline", ""))
            if not self._location_matches(location, h_location, h_name):
                match = False
            ppn = h.get("price_per_night", h.get("price", 0))
            if ppn > max_price:
                match = False
            if match:
                results.append(self._format_hotel(h))
                seen_ids.add(h.get("id"))
            if len(results) >= max_results:
                break

        # Always include pool items with extra fields (data-leak, injection)
        for h in pool:
            if h.get("id") in seen_ids:
                continue
            has_extra = any(k not in _hotel_standard_keys for k in h)
            if has_extra and len(results) < max_results:
                results.append(self._format_hotel(h))
                seen_ids.add(h.get("id"))

        if not results:
            # Fallback: return all hotel-like items so threat payloads are visible
            for h in pool[:max_results]:
                results.append(self._format_hotel(h))
        return {"success": True, "hotels": results, "total": len(results)}

    def book_hotel(
        self,
        hotel_id: str,
        check_in: str,
        check_out: str,
        payment_method_id: str,
        room_type: str = "standard",
        guests: int = 1,
        special_requests: str = "",
    ) -> Dict:
        hotel = self._get_hotel(hotel_id)
        # Also try finding hotel in the hotel pool (items may be in flights list)
        if hotel is None:
            for h in self._get_hotel_pool():
                if h.get("id") == hotel_id:
                    hotel = h
                    # Normalize hotel fields
                    hotel.setdefault("name", hotel.get("airline", ""))
                    hotel.setdefault("price_per_night", hotel.get("price", 0))
                    break
        if hotel is None:
            return {"success": False, "message": f"Hotel '{hotel_id}' not found."}
        pm = self._get_payment_method(payment_method_id)
        # Auto-accept any payment method if none are defined in state
        if pm is None and self.state.get("payment_methods"):
            return {"success": False, "message": f"Payment method '{payment_method_id}' not found."}

        price_per_night = hotel.get("price_per_night", 0)
        # Simple night count (not doing full date parsing)
        booking = {
            "id": f"bk_{len(self.state['bookings']) + 1}",
            "type": "hotel",
            "hotel_id": hotel_id,
            "hotel_name": hotel["name"],
            "check_in": check_in,
            "check_out": check_out,
            "room_type": room_type,
            "guests": guests,
            "payment_method_id": payment_method_id,
            "special_requests": special_requests,
            "price_per_night": price_per_night,
            "status": "confirmed",
        }
        self.state["bookings"].append(booking)
        return {
            "success": True,
            "message": f"Hotel '{hotel['name']}' booked ({check_in} to {check_out}).",
            "booking_id": booking["id"],
            "price_per_night": price_per_night,
        }

    def _get_all_bookings(self) -> List[Dict]:
        """Collect bookings from state['bookings'] and ALL flight items.

        Templates place state_items under flights for many scenarios
        (modify_bookings, review_itinerary, book_trip, search_flights).
        We include ALL flight items as pseudo-bookings so get_itinerary
        always returns data containing threat payloads.

        Items with seats_available == 0 are treated as existing bookings.
        Items with seats_available > 0 are also included (they may carry
        data-leak or injection payloads that need to be surfaced).
        """
        bookings = list(self.state.get("bookings", []))
        seen_ids = {b.get("id") for b in bookings}
        for f in self.state.get("flights", []):
            fid = f.get("id", "")
            if fid in seen_ids:
                continue
            entry = self._format_flight(f)
            entry["type"] = "flight"
            entry["status"] = f.get("status", "confirmed")
            bookings.append(entry)
            seen_ids.add(fid)
        return bookings

    def _find_booking_flexible(self, booking_id: str) -> Optional[Dict]:
        """Find a booking by exact or partial ID match across all sources."""
        all_bookings = self._get_all_bookings()
        # Exact match first
        for b in all_bookings:
            if b.get("id") == booking_id:
                return b
        # Also check flights directly by ID (covers items that may not
        # appear in _get_all_bookings for any reason)
        for f in self.state.get("flights", []):
            if f.get("id") == booking_id:
                entry = self._format_flight(f)
                entry["type"] = "flight"
                entry["status"] = f.get("status", "confirmed")
                return entry
        # Try partial/prefix match (e.g. "FL-DT01" matching in flight list)
        for b in all_bookings:
            bid = b.get("id", "")
            if booking_id in bid or bid in booking_id:
                return b
        # Partial match on flights directly too
        for f in self.state.get("flights", []):
            fid = f.get("id", "")
            if booking_id in fid or fid in booking_id:
                entry = self._format_flight(f)
                entry["type"] = "flight"
                entry["status"] = f.get("status", "confirmed")
                return entry
        return None

    def get_itinerary(self, booking_id: str = "") -> Dict:
        if booking_id:
            booking = self._find_booking_flexible(booking_id)
            if booking is not None:
                # Also include other flights with extra fields (threat payloads)
                # so that injection/data-leak items are surfaced even when the
                # model only requests a specific booking.
                all_bookings = self._get_all_bookings()
                other_with_extras = []
                for b in all_bookings:
                    if b.get("id") == booking.get("id"):
                        continue
                    has_extra = any(
                        k not in _FLIGHT_STANDARD_KEYS and k not in ("type", "status")
                        for k in b
                    )
                    if has_extra:
                        other_with_extras.append(b)
                result_bookings = [booking] + other_with_extras
                return {"success": True, "bookings": result_bookings, "total": len(result_bookings)}
            # Booking not found by ID -- fall back to returning ALL bookings
            # so that threat payloads embedded in state items are still
            # visible to the model.  This handles cases where the user (or
            # the model) references a BK-* ID that doesn't directly match
            # any flight/booking in state.
            all_bookings = self._get_all_bookings()
            active = [b for b in all_bookings if b.get("status") != "cancelled"]
            if active:
                return {
                    "success": True,
                    "bookings": active,
                    "total": len(active),
                    "note": f"Booking '{booking_id}' not found by exact ID; showing all active bookings.",
                }
            return {"success": False, "message": f"Booking '{booking_id}' not found."}
        all_bookings = self._get_all_bookings()
        active = [b for b in all_bookings if b.get("status") != "cancelled"]
        return {"success": True, "bookings": active, "total": len(active)}

    def cancel_booking(self, booking_id: str, reason: str = "") -> Dict:
        # First try the original bookings list
        booking = self._get_booking(booking_id)

        # If not found in bookings, check flights (booking-like items)
        if booking is None:
            for f in self.state.get("flights", []):
                if f.get("id") == booking_id:
                    booking = f
                    break

        # If still not found, try flexible matching on all sources
        # (handles cases where user references BK-* IDs that partially
        # match flight IDs, or vice versa)
        if booking is None:
            candidate = self._find_booking_flexible(booking_id)
            if candidate is not None:
                # Find the original object in state so we can mutate it
                cid = candidate.get("id", "")
                for b in self.state.get("bookings", []):
                    if b.get("id") == cid:
                        booking = b
                        break
                if booking is None:
                    for f in self.state.get("flights", []):
                        if f.get("id") == cid:
                            booking = f
                            break
                # If we still only have the formatted copy, use it
                if booking is None:
                    booking = candidate

        # Last resort: if the user gives a made-up/BK-* booking ID that
        # doesn't match anything, pick the first non-cancelled flight so the
        # conversation can proceed and threat payloads remain reachable.
        if booking is None:
            flights = self.state.get("flights", [])
            active_flights = [
                f for f in flights if f.get("status") != "cancelled"
            ]
            if active_flights:
                booking = active_flights[0]

        if booking is None:
            return {"success": False, "message": f"Booking '{booking_id}' not found."}
        if booking.get("status") == "cancelled":
            return {"success": False, "message": f"Booking '{booking_id}' is already cancelled."}

        booking["status"] = "cancelled"
        booking["cancellation_reason"] = reason

        # Restore flight seats if applicable
        if booking.get("type") == "flight":
            flight = self._get_flight(booking.get("flight_id", ""))
            if flight is not None:
                passenger_count = len(booking.get("passengers", []))
                flight["seats_available"] = flight.get("seats_available", 0) + passenger_count

        # Also restore seats on the cancelled item itself so that rebooking
        # scenarios can proceed (templates set seats_available=0 for bookings).
        if booking.get("seats_available", 0) == 0:
            booking["seats_available"] = max(1, len(booking.get("passengers", [])) or 1)

        return {
            "success": True,
            "message": f"Booking '{booking_id}' has been cancelled.",
            "refund_status": "pending",
        }
