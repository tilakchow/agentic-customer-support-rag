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
