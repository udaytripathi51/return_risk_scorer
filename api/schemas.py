"""Pydantic contracts for the scoring API (PROJECT_SPEC.md section 5).

DELIBERATE DESIGN NOTE: `category` IS A PLAIN STRING, NOT AN ENUM
------------------------------------------------------------------
Typing it as an Enum would make an unknown category a 422 at the schema boundary, which
would make the section 6.7 fail-closed path unreachable dead code. Section 6.7 requires
an unrecognised category to be *scored* as `manual_review` with an explaining reason, not
rejected as malformed -- a merchant adding a new catalogue category should get a safe,
reviewable answer, not an integration error. So validation of the category value lives in
the service layer, where the fail-closed behaviour is.

`is_prepaid` is by contrast validated here: it is definitionally `1 - is_cod`, so a
payload asserting both or neither is internally contradictory and there is no safe way to
guess the intent. That is a client bug, and a 422 naming the problem is more useful than
silently picking one. A 422 is still fail-closed -- it can never produce an
`auto_approve`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MAX_BATCH_SIZE = 100


class OrderRequest(BaseModel):
    """One order + return request to be scored."""

    customer_return_rate_lt: float = Field(
        ..., ge=0.0, le=1.0, description="Lifetime share of orders this customer returned"
    )
    customer_return_rate_90d: float = Field(
        ..., ge=0.0, le=1.0, description="Share of orders returned in the last 90 days"
    )
    customer_account_age_days: int = Field(..., ge=0, description="Account tenure in days")
    customer_order_velocity_7d: int = Field(
        ..., ge=0, description="Orders placed by this customer in the last 7 days"
    )
    order_value: float = Field(..., gt=0, description="Order value in INR")
    discount_percentage: float = Field(..., ge=0.0, le=100.0)
    category: str = Field(
        ..., description="Apparel | Electronics | Home | Beauty | Books"
    )
    category_return_base_rate: float = Field(..., ge=0.0, le=1.0)
    is_prepaid: int = Field(..., ge=0, le=1)
    is_cod: int = Field(..., ge=0, le=1)
    days_to_return: float = Field(
        ..., ge=-1.0, description="Days from delivery to return request; -1 if unknown"
    )
    same_address_returns_7d: int = Field(..., ge=0)
    same_email_returns_7d: int = Field(..., ge=0)
    payment_hash_collision: int = Field(..., ge=0, le=1)
    ip_phone_collision: int = Field(..., ge=0, le=1)
    rto_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Historical RTO rate for the delivery pincode"
    )

    @model_validator(mode="after")
    def _payment_flags_consistent(self) -> "OrderRequest":
        if self.is_prepaid + self.is_cod != 1:
            raise ValueError(
                "is_prepaid and is_cod are mutually exclusive and must sum to 1 "
                f"(got is_prepaid={self.is_prepaid}, is_cod={self.is_cod})"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_return_rate_lt": 0.62,
                "customer_return_rate_90d": 0.75,
                "customer_account_age_days": 18,
                "customer_order_velocity_7d": 5,
                "order_value": 4899.0,
                "discount_percentage": 50.0,
                "category": "Apparel",
                "category_return_base_rate": 0.26,
                "is_prepaid": 0,
                "is_cod": 1,
                "days_to_return": 13.0,
                "same_address_returns_7d": 3,
                "same_email_returns_7d": 2,
                "payment_hash_collision": 1,
                "ip_phone_collision": 1,
                "rto_risk_score": 0.44,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    value: float | None = None
    contribution: float | None = Field(
        None,
        description=(
            "SHAP contribution in the model's raw log-odds margin space. NOT on the same "
            "scale as risk_score, and null when the decision did not come from the model "
            "(e.g. a fail-closed route)."
        ),
    )
    direction: str | None = None
    reason: str


class ScoreResponse(BaseModel):
    risk_score: float = Field(..., description="Calibrated fraud probability, 0-1")
    action: Literal["auto_approve", "manual_review"]
    reasons: list[Reason]
    threshold_used: float
    model_version: str
    data_quality_notes: list[str] = Field(default_factory=list)


class BatchRequest(BaseModel):
    orders: list[OrderRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    summary: dict[str, Any]
