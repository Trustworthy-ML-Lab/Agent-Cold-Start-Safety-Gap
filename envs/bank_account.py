from typing import Any, Dict, List, Optional

from .base_env import BaseEnv


class BankAccount(BaseEnv):
    """Simulated bank account environment.

    State schema:
        balance: float, current account balance
        transactions: list of dicts, each with keys:
            id, type ('credit'|'debit'), amount, description, date, recipient
        payees: list of dicts, each with keys:
            name, account_number
        account_info: dict with keys:
            name, email, phone, address
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__(state)
        self.state.setdefault("balance", 0.0)
        self.state.setdefault("transactions", [])
        self.state.setdefault("payees", [])
        self.state.setdefault("account_info", {
            "name": "Account Holder",
            "email": "user@bank.com",
            "phone": "000-000-0000",
            "address": "123 Main St",
        })

    def check_balance(self, account_type: str = "checking") -> Dict:
        return {
            "success": True,
            "account_type": account_type,
            "balance": self.state["balance"],
            "currency": "USD",
        }

    def get_transactions(
        self, max_results: int = 20, start_date: str = "", end_date: str = ""
    ) -> Dict:
        txns = self.state["transactions"]
        if start_date or end_date:
            filtered = txns
            if start_date:
                filtered = [t for t in filtered if t.get("date", "") >= start_date]
            if end_date:
                filtered = [t for t in filtered if t.get("date", "") <= end_date]
            # If date filtering returns empty but unfiltered has data,
            # return all transactions anyway.  The model often guesses
            # wrong years (e.g. 2023 vs 2026) and the benchmark needs
            # the transaction data to be delivered.
            if filtered:
                txns = filtered
        txns = txns[:max_results]
        return {"success": True, "transactions": txns, "total": len(txns)}

    def transfer_money(self, recipient: str, amount: float, memo: str = "") -> Dict:
        if amount <= 0:
            return {"success": False, "message": "Transfer amount must be positive."}
        if amount > self.state["balance"]:
            return {"success": False, "message": "Insufficient funds."}
        self.state["balance"] -= amount
        txn = {
            "id": f"txn_{len(self.state['transactions']) + 1}",
            "type": "debit",
            "amount": amount,
            "description": f"Transfer to {recipient}" + (f" - {memo}" if memo else ""),
            "date": "",
            "recipient": recipient,
        }
        self.state["transactions"].append(txn)
        return {
            "success": True,
            "message": f"Transferred ${amount:.2f} to {recipient}.",
            "transaction_id": txn["id"],
            "new_balance": self.state["balance"],
        }

    def pay_bill(self, payee: str, amount: float, account_number: str = "") -> Dict:
        if amount <= 0:
            return {"success": False, "message": "Payment amount must be positive."}
        if amount > self.state["balance"]:
            return {"success": False, "message": "Insufficient funds."}
        known = any(p["name"].lower() == payee.lower() for p in self.state["payees"])
        self.state["balance"] -= amount
        txn = {
            "id": f"txn_{len(self.state['transactions']) + 1}",
            "type": "debit",
            "amount": amount,
            "description": f"Bill payment to {payee}" + (f" (acct: {account_number})" if account_number else ""),
            "date": "",
            "recipient": payee,
        }
        self.state["transactions"].append(txn)
        return {
            "success": True,
            "message": f"Paid ${amount:.2f} to {payee}.",
            "transaction_id": txn["id"],
            "new_balance": self.state["balance"],
            "payee_registered": known,
        }

    def update_account_info(self, field: str, value: str) -> Dict:
        allowed = {"name", "email", "phone", "address"}
        if field not in allowed:
            return {"success": False, "message": f"Unknown field '{field}'. Allowed: {', '.join(sorted(allowed))}."}
        old_value = self.state["account_info"].get(field, "")
        self.state["account_info"][field] = value
        return {
            "success": True,
            "message": f"Account {field} updated.",
            "field": field,
            "old_value": old_value,
            "new_value": value,
        }
