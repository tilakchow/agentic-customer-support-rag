"""Unit and integration tests for Item 10: Operational Tools and plan_tools node."""

import pytest
from langchain_core.messages import HumanMessage

from agenticrag.nodes import plan_tools
from agenticrag.schemas import ToolCallRequest, ToolPlan
from agenticrag.tools import create_support_ticket, get_order_status, reset_tools_db


@pytest.fixture(autouse=True)
def clean_tools_database():
    reset_tools_db()
    yield
    reset_tools_db()


def test_get_order_status_success_and_fields():
    res = get_order_status("ORD-1001", tenant_id="default")
    assert res["found"] is True
    assert res["order_id"] == "ORD-1001"
    assert res["status"] == "shipped"
    assert res["carrier"] == "FedEx"
    assert "Wireless Noise-Canceling Headphones" in res["item_description"]


def test_get_order_status_not_found():
    res = get_order_status("ORD-9999", tenant_id="default")
    assert res["found"] is False
    assert "not found" in res["error"]


def test_order_status_tenant_isolation():
    # ORD-2001 is seeded under tenant-acme
    res_default = get_order_status("ORD-2001", tenant_id="default")
    assert res_default["found"] is False

    res_acme = get_order_status("ORD-2001", tenant_id="tenant-acme")
    assert res_acme["found"] is True
    assert res_acme["status"] == "delivered"


def test_create_support_ticket():
    res = create_support_ticket("Late delivery investigation", order_id="ORD-1001")
    assert res["success"] is True
    assert res["ticket_id"].startswith("TICK-")
    assert res["order_id"] == "ORD-1001"
    assert res["status"] == "open"


@pytest.mark.asyncio
async def test_plan_tools_extracts_order_lookup():
    state = {
        "messages": [HumanMessage(content="What is the delivery status for ORD-1002?")],
        "tenant_id": "default",
    }
    res = await plan_tools(state)
    calls = res["tool_calls"]
    results = res["tool_results"]

    assert len(calls) == 1
    assert calls[0]["tool_name"] == "get_order_status"
    assert calls[0]["order_id"] == "ORD-1002"

    assert len(results) == 1
    assert results[0]["result"]["status"] == "delivered"


@pytest.mark.asyncio
async def test_plan_tools_enforces_max_two_calls():
    # Force a state with multiple order references
    state = {
        "messages": [
            HumanMessage(
                content="Please check ORD-1001, ORD-1002, ORD-1003 and open a ticket for damaged item"
            )
        ],
        "tenant_id": "default",
    }
    res = await plan_tools(state)
    calls = res["tool_calls"]
    assert len(calls) <= 2


@pytest.mark.asyncio
async def test_plan_tools_ignores_general_queries():
    state = {
        "messages": [HumanMessage(content="How do I change my account password?")],
        "tenant_id": "default",
    }
    res = await plan_tools(state)
    assert res["tool_calls"] == []
    assert res["tool_results"] == []


def test_tool_plan_pydantic_schema_validation():
    req = ToolCallRequest(tool_name="get_order_status", order_id="ORD-1001")
    plan = ToolPlan(needs_tools=True, calls=[req])
    assert plan.needs_tools is True
    assert len(plan.calls) == 1
    assert plan.calls[0].tool_name == "get_order_status"
