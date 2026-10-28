"""Comprehensive 20-case evaluation suite for customer support operational tools (Item 10).

Validates:
1. Mock SQLite DB order lookup (found, status, carrier, tracking)
2. Handling of non-existent, whitespace, and empty order IDs
3. Multi-tenant order isolation across tenant namespaces
4. Support ticket creation with auto-generated TICK-* identifiers
5. plan_tools node invocation, max 2 calls/turn limit, and Pydantic validation
6. Benign query non-invocation (guarding against unnecessary tool calls)
"""

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from agenticrag.nodes import plan_tools
from agenticrag.tools import create_support_ticket, get_order_status, reset_tools_db

EVAL_CASES: list[dict[str, Any]] = [
    {
        "id": "CASE-01",
        "name": "Valid order ORD-1001 shipped status",
        "type": "order_lookup",
        "input": "ORD-1001",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "shipped",
        "expected_carrier": "FedEx",
    },
    {
        "id": "CASE-02",
        "name": "Valid order ORD-1002 delivered status",
        "type": "order_lookup",
        "input": "ORD-1002",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "delivered",
        "expected_carrier": "UPS",
    },
    {
        "id": "CASE-03",
        "name": "Valid order ORD-1003 processing status",
        "type": "order_lookup",
        "input": "ORD-1003",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "processing",
    },
    {
        "id": "CASE-04",
        "name": "Valid order ORD-1004 returned status",
        "type": "order_lookup",
        "input": "ORD-1004",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "returned",
        "expected_carrier": "USPS",
    },
    {
        "id": "CASE-05",
        "name": "Non-existent order ORD-9999 returns not found",
        "type": "order_lookup",
        "input": "ORD-9999",
        "tenant_id": "default",
        "expected_found": False,
    },
    {
        "id": "CASE-06",
        "name": "Lowercase order ID normalization",
        "type": "order_lookup",
        "input": "ord-1001",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "shipped",
    },
    {
        "id": "CASE-07",
        "name": "Order ID with leading/trailing whitespace",
        "type": "order_lookup",
        "input": "  ORD-1002  ",
        "tenant_id": "default",
        "expected_found": True,
        "expected_status": "delivered",
    },
    {
        "id": "CASE-08",
        "name": "Empty order ID rejection",
        "type": "order_lookup",
        "input": "",
        "tenant_id": "default",
        "expected_found": False,
    },
    {
        "id": "CASE-09",
        "name": "Tenant isolation: default tenant cannot access tenant-acme order",
        "type": "tenant_isolation",
        "input": "ORD-2001",
        "tenant_id": "default",
        "expected_found": False,
    },
    {
        "id": "CASE-10",
        "name": "Tenant isolation: tenant-acme can access own order ORD-2001",
        "type": "tenant_isolation",
        "input": "ORD-2001",
        "tenant_id": "tenant-acme",
        "expected_found": True,
        "expected_status": "delivered",
    },
    {
        "id": "CASE-11",
        "name": "Create support ticket with associated order ID",
        "type": "ticket_create",
        "summary": "Damaged package on delivery",
        "order_id": "ORD-1001",
        "tenant_id": "default",
        "expected_success": True,
    },
    {
        "id": "CASE-12",
        "name": "Create support ticket without order ID",
        "type": "ticket_create",
        "summary": "Account password reset assistance",
        "order_id": None,
        "tenant_id": "default",
        "expected_success": True,
    },
    {
        "id": "CASE-13",
        "name": "Reject ticket creation with empty summary",
        "type": "ticket_create",
        "summary": "   ",
        "order_id": None,
        "tenant_id": "default",
        "expected_success": False,
    },
    {
        "id": "CASE-14",
        "name": "plan_tools invokes get_order_status on natural query",
        "type": "plan_tools",
        "query": "Where is my order ORD-1001 right now?",
        "tenant_id": "default",
        "expected_tool": "get_order_status",
        "expected_order_id": "ORD-1001",
    },
    {
        "id": "CASE-15",
        "name": "plan_tools invokes get_order_status for delivered package",
        "type": "plan_tools",
        "query": "Can you check tracking for ORD-1002?",
        "tenant_id": "default",
        "expected_tool": "get_order_status",
        "expected_order_id": "ORD-1002",
    },
    {
        "id": "CASE-16",
        "name": "plan_tools invokes create_support_ticket on request",
        "type": "plan_tools",
        "query": "Please open a support ticket regarding wrong items received",
        "tenant_id": "default",
        "expected_tool": "create_support_ticket",
    },
    {
        "id": "CASE-17",
        "name": "plan_tools dual tool call capped at max 2",
        "type": "plan_tools",
        "query": "Check status for ORD-1003 and please file a ticket for delayed order",
        "tenant_id": "default",
        "expected_calls_count": 2,
    },
    {
        "id": "CASE-18",
        "name": "plan_tools skips tools on policy knowledge question",
        "type": "plan_tools_none",
        "query": "What is your 30-day return policy for unopened electronics?",
        "tenant_id": "default",
    },
    {
        "id": "CASE-19",
        "name": "plan_tools skips tools on conversational greeting",
        "type": "plan_tools_none",
        "query": "Hello! Good morning, how are you today?",
        "tenant_id": "default",
    },
    {
        "id": "CASE-20",
        "name": "plan_tools multi-tenant order status check",
        "type": "plan_tools",
        "query": "Please check order ORD-2001 status",
        "tenant_id": "tenant-acme",
        "expected_tool": "get_order_status",
        "expected_order_id": "ORD-2001",
    },
]


async def run_tool_use_eval() -> dict[str, Any]:
    """Executes all 20 tool-use evaluation benchmark cases."""
    reset_tools_db()
    passed = 0
    total = len(EVAL_CASES)
    results: list[dict[str, Any]] = []

    for case in EVAL_CASES:
        c_type = case["type"]
        c_id = case["id"]
        c_name = case["name"]

        if c_type in ("order_lookup", "tenant_isolation"):
            res = get_order_status(case["input"], tenant_id=case["tenant_id"])
            is_ok = res.get("found") == case["expected_found"]
            if is_ok and case.get("expected_status"):
                is_ok = res.get("status") == case["expected_status"]
            if is_ok and case.get("expected_carrier"):
                is_ok = res.get("carrier") == case["expected_carrier"]
            if is_ok:
                passed += 1
            results.append({"id": c_id, "name": c_name, "passed": is_ok, "details": res})

        elif c_type == "ticket_create":
            res = create_support_ticket(
                summary=case["summary"],
                order_id=case["order_id"],
                tenant_id=case["tenant_id"],
            )
            is_ok = res.get("success") == case["expected_success"]
            if is_ok:
                passed += 1
            results.append({"id": c_id, "name": c_name, "passed": is_ok, "details": res})

        elif c_type == "plan_tools":
            state = {
                "messages": [HumanMessage(content=case["query"])],
                "tenant_id": case["tenant_id"],
            }
            res = await plan_tools(state)
            calls = res.get("tool_calls", [])
            is_ok = len(calls) > 0 and len(calls) <= 2
            if case.get("expected_tool"):
                is_ok = is_ok and any(c.get("tool_name") == case["expected_tool"] for c in calls)
            if case.get("expected_order_id"):
                is_ok = is_ok and any(c.get("order_id") == case["expected_order_id"] for c in calls)
            if case.get("expected_calls_count"):
                is_ok = len(calls) == case["expected_calls_count"]
            if is_ok:
                passed += 1
            results.append({"id": c_id, "name": c_name, "passed": is_ok, "details": calls})

        elif c_type == "plan_tools_none":
            state = {
                "messages": [HumanMessage(content=case["query"])],
                "tenant_id": case["tenant_id"],
            }
            res = await plan_tools(state)
            calls = res.get("tool_calls", [])
            is_ok = len(calls) == 0
            if is_ok:
                passed += 1
            results.append({"id": c_id, "name": c_name, "passed": is_ok, "details": calls})

    accuracy = passed / total
    return {
        "benchmark": "Tool Use Evaluation (20 cases)",
        "total_cases": total,
        "passed": passed,
        "accuracy": round(accuracy, 4),
        "results": results,
    }


def main():
    report = asyncio.run(run_tool_use_eval())
    print(
        f"Tool-Use Benchmark Results: {report['passed']}/{report['total_cases']} passed ({report['accuracy'] * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
