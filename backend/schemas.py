# backend/schemas.py
# ==============================================================================
# Modelos Pydantic para structured output de los LLMs del grafo RAG.
# - IntentClassification: salida del clasificador de intencion (nodo classify_intent)
# ==============================================================================
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Mapa de secuencias corruptas comunes a sus caracteres correctos ──────────
# Patron: caracter de reemplazo Unicode (U+FFFD) seguido de un codigo hex
# Ejemplo: \ufffd f3 → ó, \ufffd e1 → á, \ufffd ed → í
_MOJIBAKE_MAP = {
    "\ufffde1": "á", "\ufffde9": "é", "\ufffded": "í",
    "\ufffdf3": "ó", "\ufffdfa": "ú", "\ufffdf1": "ñ",
    "\ufffdc1": "Á", "\ufffdc9": "É", "\ufffdcd": "Í",
    "\ufffdd3": "Ó", "\ufffdda": "Ú", "\ufffdd1": "Ñ",
    "\ufffdfc": "ü", "\ufffddc": "Ü",
}
_MOJIBAKE_RE = re.compile("|".join(re.escape(k) for k in _MOJIBAKE_MAP))


def _fix_encoding(text: str) -> str:
    """Corrige caracteres corruptos (mojibake) en texto generado por LLMs."""
    if not text or "\ufffd" not in text:
        return text
    return _MOJIBAKE_RE.sub(lambda m: _MOJIBAKE_MAP[m.group()], text)


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
        suggested_k:             Numero de chunks sugerido por el LLM (1-20), None para GENERAL
    """

    intent: Literal["Búsqueda", "Resumen", "Comparación", "GENERAL"]
    needs_retrieval: bool
    reason: str = Field(min_length=1, max_length=240)
    entities: IntentEntities
    suggested_k: Optional[int] = Field(default=None, ge=1, le=20)


def intent_to_dict(intent: IntentClassification) -> dict[str, Any]:
    """Convierte IntentClassification a dict para almacenar en RAGState.

    Aplica _fix_encoding a campos de texto para corregir mojibake.
    """
    data = intent.model_dump()
    if data.get("reason"):
        data["reason"] = _fix_encoding(data["reason"])
    return data


