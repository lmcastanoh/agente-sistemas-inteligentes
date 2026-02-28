# backend/schemas.py
# ==============================================================================
# Modelos Pydantic para structured output de los LLMs del grafo RAG.
# - IntentClassification: salida del clasificador de intencion (nodo classify_intent)
# - GroundingEvaluation: salida del critico de grounding (nodo evaluate_grounding)
# ==============================================================================
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class IntentEntities(BaseModel):
    """Entidades extraidas de la consulta del usuario por el clasificador.

    Campos:
        make:  Marca del vehiculo (ej: 'Toyota', 'Mazda'). None si no se menciona.
        model: Modelo del vehiculo (ej: 'Hilux', 'CX-5'). None si no se menciona.
        year:  Ano del modelo (ej: '2025'). None si no se menciona.
        trim:  Version o trim (ej: 'SR5', 'GR Sport'). None si no se menciona.
    """

    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[str] = None
    trim: Optional[str] = None


class IntentClassification(BaseModel):
    """Resultado de la clasificacion de intencion del usuario.

    El clasificador (gpt-5-nano) analiza la consulta y devuelve esta estructura
    que determina la ruta del grafo: GENERAL, Busqueda, Resumen o Comparacion.

    Campos:
        intent:                  Categoria clasificada (Busqueda|Resumen|Comparacion|GENERAL)
        needs_retrieval:         True si la consulta requiere busqueda en ChromaDB
        reason:                  Explicacion corta de por que se eligio ese intent
        entities:                Marca, modelo, ano y version extraidos de la consulta
        clarification_question:  Pregunta de aclaracion si falta informacion (ej: ano/version)
        suggested_k:             Numero de chunks sugerido por el LLM (1-20), None para GENERAL
    """

    intent: Literal["Búsqueda", "Resumen", "Comparación", "GENERAL"]
    needs_retrieval: bool
    reason: str = Field(min_length=1, max_length=240)
    entities: IntentEntities
    clarification_question: Optional[str] = None
    suggested_k: Optional[int] = Field(default=None, ge=1, le=20)


class GroundingEvaluation(BaseModel):
    """Resultado de la evaluacion del critico de grounding.

    El critico (gpt-5-nano) evalua si la respuesta generada cumple con las
    reglas de grounding: soportada por contexto, con citas, y completa.

    Campos:
        approved:               True si la respuesta pasa la evaluacion
        score:                  Puntuacion de calidad (0.0 = muy mala, 1.0 = perfecta)
        supported_by_context:   True si toda la informacion proviene del contexto
        has_citations:          True si incluye citas en formato [doc_id=...; pagina=...]
        complete_enough:        True si la respuesta es suficientemente completa
        issues:                 Lista de problemas detectados (usada como feedback en reintentos)
        clarification_question: Pregunta sugerida si la respuesta es insuficiente
    """

    approved: bool
    score: float = Field(ge=0.0, le=1.0)
    supported_by_context: bool
    has_citations: bool
    complete_enough: bool
    issues: list[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None


def intent_to_dict(intent: IntentClassification) -> dict[str, Any]:
    """Convierte IntentClassification a dict para almacenar en RAGState."""
    return intent.model_dump()


def eval_to_dict(result: GroundingEvaluation) -> dict[str, Any]:
    """Convierte GroundingEvaluation a dict para almacenar en RAGState."""
    return result.model_dump()
