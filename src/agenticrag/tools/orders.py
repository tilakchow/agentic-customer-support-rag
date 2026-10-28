"""SQLite orders database and support ticket tools with tenant isolation."""

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

_DB_LOCK = threading.Lock()
_DB_CONN: sqlite3.Connection | None = None


def get_orders_db() -> sqlite3.Connection:
    """Returns singleton thread-safe SQLite connection initialized with seeded fake orders."""
    global _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is None:
            _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)
            _DB_CONN.row_factory = sqlite3.Row
            _init_db(_DB_CONN)
        return _DB_CONN


def _init_db(conn: sqlite3.Connection) -> None:
    """Creates schema and seeds initial fake orders and ticket records."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            status TEXT NOT NULL,
            item_description TEXT NOT NULL,
            tracking_number TEXT,
            carrier TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            order_id TEXT,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    seed_orders = [
        (
            "ORD-1001",
            "default",
            "CUST-101",
            "shipped",
            "Wireless Noise-Canceling Headphones",
            "TRK-FEDEX-987654",
            "FedEx",
            "2026-10-25T10:00:00Z",
        ),
        (
            "ORD-1002",
            "default",
            "CUST-102",
            "delivered",
            "USB-C Multi-Port Fast Hub",
            "TRK-UPS-123456",
            "UPS",
            "2026-10-20T14:30:00Z",
        ),
        (
            "ORD-1003",
            "default",
            "CUST-103",
            "processing",
            "Ergonomic Mechanical Keyboard (RGB)",
            None,
            None,
            "2026-10-27T08:15:00Z",
        ),
        (
            "ORD-1004",
            "default",
            "CUST-104",
            "returned",
            "Adjustable Aluminum Laptop Stand",
            "TRK-USPS-445566",
            "USPS",
            "2026-10-15T11:00:00Z",
        ),
        (
            "ORD-2001",
            "tenant-acme",
            "CUST-201",
            "delivered",
            "Enterprise Server Rack Unit",
            "TRK-DHL-998877",
            "DHL",
            "2026-10-22T09:00:00Z",
        ),
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO orders (order_id, tenant_id, customer_id, status, item_description, tracking_number, carrier, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        seed_orders,
    )
    conn.commit()


def get_order_status(order_id: str, tenant_id: str = "default") -> dict[str, Any]:
    """Retrieves order status, shipping carrier, and tracking details for a specific order.

    Enforces tenant isolation: orders from other non-default tenants cannot be accessed.
    """
    if not order_id or not order_id.strip():
        return {
            "found": False,
            "order_id": "",
            "error": "Order ID must not be empty.",
        }

    norm_id = order_id.strip().upper()
    conn = get_orders_db()
    with _DB_LOCK:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ? AND (tenant_id = ? OR tenant_id = 'default')",
            (norm_id, tenant_id.strip()),
        )
        row = cursor.fetchone()
        if not row:
            return {
                "found": False,
                "order_id": norm_id,
                "error": f"Order '{norm_id}' was not found for this account/tenant.",
            }
        return {
            "found": True,
            "order_id": row["order_id"],
            "status": row["status"],
            "item_description": row["item_description"],
            "tracking_number": row["tracking_number"],
            "carrier": row["carrier"],
            "created_at": row["created_at"],
        }


def create_support_ticket(
    summary: str, order_id: str | None = None, tenant_id: str = "default"
) -> dict[str, Any]:
    """Creates a new support ticket in the support ticketing system.

    Args:
        summary: A brief description of the issue or support request.
        order_id: Optional associated order ID.
        tenant_id: The tenant identifier.
    """
    if not summary or not summary.strip():
        return {"success": False, "error": "Ticket summary cannot be empty."}

    ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"
    now_iso = datetime.now(UTC).isoformat()
    norm_order = order_id.strip().upper() if order_id and order_id.strip() else None

    conn = get_orders_db()
    with _DB_LOCK:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO support_tickets (ticket_id, tenant_id, order_id, summary, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (ticket_id, tenant_id.strip(), norm_order, summary.strip(), "open", now_iso),
        )
        conn.commit()

    return {
        "success": True,
        "ticket_id": ticket_id,
        "order_id": norm_order,
        "summary": summary.strip(),
        "status": "open",
        "created_at": now_iso,
    }


def reset_tools_db() -> None:
    """Helper for test cleanup to re-initialize SQLite in-memory tables."""
    global _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            _DB_CONN.close()
            _DB_CONN = None
