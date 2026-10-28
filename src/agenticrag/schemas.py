"""Pydantic schemas for structured LLM routing and evaluation decisions."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        default="kb",
        description="Use kb for questions needing knowledge base retrieval; direct for greetings/simple chat.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_route_aliases(cls, data):
        if isinstance(data, dict) and "route" not in data:
            for key in ["decision", "destination", "choice", "answer"]:
                if key in data and data[key] in ("kb", "direct"):
                    data["route"] = data[key]
                    break
        return data


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        default="weak",
        description="good means evidence can answer the question; weak means not enough evidence.",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_grade_aliases(cls, data):
        if isinstance(data, dict) and "grade" not in data:
            for key in ["answer", "score", "decision", "result"]:
                if key in data and data[key] in ("good", "weak"):
                    data["grade"] = data[key]
                    break
        return data


class ScopeDecision(BaseModel):
    scope: Literal["in_scope", "out_of_scope", "needs_human", "policy_sensitive"] = Field(
        default="in_scope",
        description="in_scope for customer support (orders, account, returns, shipping, products, billing, cancellation); out_of_scope for unrelated inquiries (coding, trivia, homework, poems, math); needs_human when customer explicitly asks for human/agent or reports complex unresolved issues; policy_sensitive for legal threats, lawsuits, harassment, or severe complaints.",
    )
    reason: str = Field(default="", description="Brief reason for scope classification.")

    @model_validator(mode="before")
    @classmethod
    def handle_scope_aliases(cls, data):
        if isinstance(data, dict) and "scope" not in data:
            for key in ["decision", "category", "classification", "result"]:
                if key in data and data[key] in (
                    "in_scope",
                    "out_of_scope",
                    "needs_human",
                    "policy_sensitive",
                ):
                    data["scope"] = data[key]
                    break
        return data


class ClaimSupportCheck(BaseModel):
    supported: bool = Field(
        default=True,
        description="True if all claims made in the answer are supported by the provided context; False if there are fabricated or unsupported facts.",
    )
    reason: str = Field(default="", description="Explanation of claim support check.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Specific unsupported or hallucinated statements."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_support_aliases(cls, data):
        if isinstance(data, dict) and "supported" not in data:
            for key in ["grounded", "is_supported", "valid", "pass"]:
                if key in data and isinstance(data[key], bool):
                    data["supported"] = data[key]
                    break
        return data


class ToolCallRequest(BaseModel):
    tool_name: Literal["get_order_status", "create_support_ticket"] = Field(
        description="Name of the customer support tool to execute."
    )
    order_id: str | None = Field(
        default=None,
        description="Normalized order ID (e.g. ORD-1001) if checking order status or creating an order-specific ticket.",
    )
    summary: str | None = Field(
        default=None,
        description="Issue summary description if creating a support ticket.",
    )


class ToolPlan(BaseModel):
    needs_tools: bool = Field(
        default=False,
        description="True if external operational tools (order tracking or support ticket creation) are needed.",
    )
    calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="List of tool requests to execute (maximum 2 per turn).",
    )
