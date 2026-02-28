from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class IntentEntities(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[str] = None
    trim: Optional[str] = None


class IntentClassification(BaseModel):
    intent: Literal["Búsqueda", "Resumen", "Comparación", "GENERAL"]
    needs_retrieval: bool
    reason: str = Field(min_length=1, max_length=240)
    entities: IntentEntities
    clarification_question: Optional[str] = None
    suggested_k: Optional[int] = Field(default=None, ge=1, le=20)


class GroundingEvaluation(BaseModel):
    approved: bool
    score: float = Field(ge=0.0, le=1.0)
    supported_by_context: bool
    has_citations: bool
    complete_enough: bool
    issues: list[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None


def intent_to_dict(intent: IntentClassification) -> dict[str, Any]:
    return intent.model_dump()


def eval_to_dict(result: GroundingEvaluation) -> dict[str, Any]:
    return result.model_dump()
