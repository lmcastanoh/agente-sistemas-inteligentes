# RAG Agentico - Fichas Tecnicas Vehiculares

Sistema de Generacion Aumentada por Recuperacion (RAG) **agentico** especializado en fichas tecnicas
de vehiculos. Usa agentes ReAct con razonamiento autonomo para consultar, resumir y comparar
especificaciones de multiples marcas y modelos usando lenguaje natural.

## Stack Tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend API | FastAPI + Uvicorn |
| Orquestacion | LangGraph (grafo de estados con agentes ReAct) |
| LLM | OpenAI `gpt-5-nano` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Base vectorial | ChromaDB (persistencia local) |
| Frontend | Streamlit |
| Streaming | SSE token-by-token via `astream_events` |
| OCR | EasyOCR (paginas escaneadas) |
| Extraccion PDF | pdfplumber |

---

## Estructura del Proyecto

```
agente-sistemas-inteligentes/
├── backend/
│   ├── app.py              # API FastAPI: /ingest y /chat/stream (SSE streaming real)
│   ├── rag_graph.py         # Grafo LangGraph: 6 nodos con agentes ReAct
│   ├── rag_store.py         # ChromaDB: ingestion, embeddings, vector store
│   ├── tools.py             # 8 tools (buscar, comparar, resumir, refinar, corregir, regenerar)
│   ├── schemas.py           # Modelos Pydantic (IntentClassification)
│   ├── prompts.py           # System prompts (clasificador, agente ReAct, generador, evaluador)
│   ├── test_intent_routes.py # Tests de clasificacion de intent
│   ├── data/                # PDFs organizados por marca
│   │   ├── Toyota/
│   │   ├── Mazda/
│   │   ├── Volkswagen/
│   │   ├── Peugeot/
│   │   ├── Opel/
│   │   ├── MG Emotor/
│   │   └── Seat/
│   └── chroma_db/           # Base vectorial persistida (SQLite + HNSW)
├── frontend/
│   └── streamlit_app.py     # Interfaz de chat con streaming y trazabilidad
├── .env                     # Variables de entorno (OPENAI_API_KEY, HF_TOKEN)
├── requeriments.txt         # Dependencias Python
└── README.md
```

---

## Datos Disponibles

- **7 marcas**: Toyota, Mazda, Volkswagen, Peugeot, Opel, MG Emotor, Seat
- **50 modelos** indexados
- **584 chunks** en ChromaDB (chunks de 1000 chars, overlap 150)
- Metadata por chunk: `source`, `page`, `marca`, `modelo`, `doc_id`, `chunk_id`, `ocr`

---

## Arquitectura del Grafo RAG Agentico

El sistema usa un grafo LangGraph con **6 nodos** y **2 agentes ReAct**:

```
START
  │
  ▼
┌─────────────────────┐
│  1. classify_intent  │  LLM clasifica en 4 intents + extrae entidades
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────┐
    │                             │
 GENERAL                    needs_retrieval
    │                             │
    ▼                             ▼
┌──────────────┐       ┌──────────────────┐
│ answer_general│       │   2. retrieve     │  Busqueda semantica en ChromaDB
└──────┬───────┘       └────────┬─────────┘
       │                        │
       ▼                        ▼
      END              ┌──────────────────┐
                       │  3. agent_reason  │  Agente ReAct: razona y usa tools
                       │  (loop autonomo)  │  en ciclo reason → act → observe
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────────┐
                       │ 4. generate_grounded │  Genera respuesta con citas
                       │  (streaming async)   │  desde contexto + output agente
                       └────────┬─────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  5. eval_agent    │  Agente evaluador: verifica y
                       │  (loop autonomo)  │  corrige si hay problemas
                       └────────┬─────────┘
                                │
                                ▼
                               END
```

### Rutas del grafo

| Ruta | Intent | Flujo |
|------|--------|-------|
| **A** | GENERAL | classify → answer_general → END |
| **B** | Busqueda / Resumen / Comparacion | classify → retrieve → agent_reason → generate → eval_agent → END |

---

## Detalle de cada nodo

### 1. `classify_intent` — Clasificador de intencion
- **LLM**: gpt-5-nano (temperature=0, structured output)
- Clasifica en: Busqueda, Resumen, Comparacion, GENERAL
- Extrae entidades: marca, modelo, ano, version
- Sugiere `suggested_k` (cuantos chunks recuperar)
- Memory: si no hay modelo en la pregunta, usa `last_model`/`last_make` del turno anterior
- Keyword fallback: regex corrige clasificaciones erroneas

### 2. `retrieve` — Recuperacion semantica
- Busca en ChromaDB con `similarity_search`
- **Dynamic k**: usa `suggested_k` del clasificador o mapa fijo por intent
- Reescribe la query para resolver referencias anaforicas ("ese modelo" → nombre real)
- Filtros de metadata por marca/modelo con variantes normalizadas
- Comparaciones: retrieval balanceado (k/2 por cada modelo)

### 3. `agent_reason` — Agente ReAct (razonamiento autonomo)
- **LLM**: gpt-5-nano (temperature=0.2, tools bindeadas)
- **Ciclo ReAct**: reason → act (tool_call) → observe → reason (max 3 iteraciones)
- Decide autonomamente que tools usar segun el contexto
- Auto-reflexion integrada: verifica si el contexto es suficiente antes de responder
- Verificacion de relevancia: no mezcla datos de modelos diferentes
- Si el contexto es insuficiente, usa `refinar_busqueda` con query reformulada

### 4. `generate_grounded` — Generacion con grounding (async streaming)
- **LLM**: gpt-5-nano (temperature=0.2, async con `astream`)
- Combina contexto de retrieval + output del agente ReAct
- Citas como referencias numeradas [1], [2] con seccion **Fuentes** al final
- Anti-hallucination: verifica que los chunks correspondan al modelo pedido
- Streaming token-by-token hacia el frontend

### 5. `eval_agent` — Agente evaluador de grounding
- **LLM**: gpt-5-nano (temperature=0, tools bindeadas)
- **Ciclo ReAct**: evalua → corrige si hay problemas (max 2 iteraciones)
- Si la respuesta es buena, la pasa tal cual (sin cambios)
- Si hay problemas menores (formato, citas): usa `corregir_respuesta`
- Si hay problemas graves (modelo equivocado, datos inventados): usa `refinar_busqueda` + `regenerar_respuesta`

---

## Tools disponibles (8)

| Tool | Descripcion | Usada por |
|------|-------------|-----------|
| `listar_modelos_disponibles` | Catalogo de modelos indexados en ChromaDB | agent_reason |
| `buscar_especificacion` | Dato tecnico puntual (potencia, torque, etc.) | agent_reason |
| `buscar_por_marca` | Todos los modelos de una marca | agent_reason |
| `comparar_modelos` | Tabla comparativa markdown entre 2 modelos | agent_reason |
| `resumir_ficha` | Resumen estructurado de ficha tecnica | agent_reason |
| `refinar_busqueda` | Busqueda adicional con query/filtros diferentes | agent_reason, eval_agent |
| `corregir_respuesta` | Corrige formato, citas, datos no respaldados | eval_agent |
| `regenerar_respuesta` | Regenera respuesta completa desde contexto mejorado | eval_agent |

---

## Mecanismos anti-hallucination

1. **Verificacion de modelo**: el agente ReAct y el generador verifican que los chunks correspondan al modelo pedido
2. **Citas obligatorias**: referencias numeradas [1], [2] con seccion Fuentes
3. **Evaluador agentico**: `eval_agent` verifica la respuesta y corrige si hay problemas
4. **No-mezcla de modelos**: si el contexto tiene datos de otros modelos, los descarta
5. **Declaracion explicita**: si no hay datos, dice "No se encontro informacion" en vez de inventar
6. **Keyword fallback**: regex corrige clasificaciones erroneas del LLM

---

## Streaming y trazabilidad

### Streaming SSE token-by-token
- El backend usa `astream_events` para emitir tokens en tiempo real
- El frontend muestra texto apareciendo progresivamente
- Indicadores de progreso por nodo: "Clasificando intencion...", "Buscando documentos...", etc.

### Eventos SSE
| Evento | Contenido |
|--------|-----------|
| `token` | Token individual de la respuesta (con newlines escapados) |
| `progress` | Nombre del nodo que acaba de terminar |
| `trazabilidad` | JSON con ruta, clasificacion, chunks, pasos del agente, evaluacion |
| `done` | Fin del stream |

### Trazabilidad
Cada respuesta incluye un panel expandible con:
- Ruta del grafo (nodos ejecutados)
- Intencion clasificada y si requiere RAG
- Chunks recuperados (fuente, pagina, k utilizado)
- Pasos del agente ReAct (tools invocadas y argumentos)
- Evaluacion del `eval_agent` (aprobada o corregida)

---

## Configuracion en Windows

### Requisitos previos

- **Python 3.12** — descargar de https://www.python.org/downloads/
- **Git** — descargar de https://git-scm.com/
- Cuenta de OpenAI con API key activa (para gpt-5-nano)

### 1. Clonar el repositorio

```powershell
git clone https://github.com/lmcastanoh/agente-sistemas-inteligentes.git
cd agente-sistemas-inteligentes
```

### 2. Crear entorno virtual

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
```

### 3. Instalar dependencias

```powershell
pip install -r requeriments.txt
```

### 4. Configurar variables de entorno

Crear archivo `.env` en la raiz del proyecto:

```env
OPENAI_API_KEY=sk-tu-clave-aqui
HF_TOKEN=hf-tu-token-aqui
```

### 5. Agregar documentos PDF

Colocar los PDFs dentro de `backend/data/` organizados por marca:

```
backend/data/
├── Toyota/
│   ├── ficha-tecnica-hilux.pdf
│   └── ficha-tecnica-fortuner.pdf
├── Mazda/
│   └── ficha-tecnica-mazda-cx-5-2026.pdf
└── ...
```

### 6. Ejecutar el backend (Terminal 1)

```powershell
cd agente-sistemas-inteligentes
.\.venv\Scripts\Activate
cd backend
uvicorn app:app --reload --port 8001
```

Verificar en: http://localhost:8001/docs

### 7. Ingestar documentos

Desde Swagger UI (http://localhost:8001/docs) o con curl:

```powershell
curl -X POST http://localhost:8001/ingest -H "Content-Type: application/json" -d "{\"data_dir\": \"./data\"}"
```

Respuesta esperada:

```json
{
  "files_dir": "./data",
  "raw_docs": 50,
  "chunks": 584,
  "ids_added": 584
}
```

### 8. Ejecutar el frontend (Terminal 2)

```powershell
cd agente-sistemas-inteligentes
.\.venv\Scripts\Activate
cd frontend
streamlit run streamlit_app.py
```

Acceder en: http://localhost:8501

---

## Endpoints de la API

### `POST /ingest`

Ingesta documentos PDF desde un directorio.

```json
// Request
{"data_dir": "./data"}

// Response
{"files_dir": "./data", "raw_docs": 50, "chunks": 584, "ids_added": 584}
```

### `POST /chat/stream`

Chat con streaming SSE token-by-token.

```json
// Request
{"question": "¿Cual es la potencia del Toyota Hilux?", "session_id": "sesion-1"}
```

---

## Ejemplos de preguntas

| Tipo | Ejemplo |
|------|---------|
| Busqueda puntual | "¿Cual es la potencia del Toyota Hilux 2024?" |
| Resumen | "Resume la ficha tecnica del Mazda CX-5" |
| Comparacion | "Compara el Hilux vs el Fortuner" |
| General | "¿Que es el torque?" |
| Por marca | "¿Que modelos de Volkswagen tienen?" |
| Refinamiento | "¿Y cuanto pesa?" (follow-up del modelo anterior) |

---

## Solucion de problemas

### Puerto en uso

```powershell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

### Error de OpenAI 401

Verificar que `OPENAI_API_KEY` esta configurada en `.env`.

### Error de OpenAI 429 (cuota excedida)

Verificar creditos en: https://platform.openai.com/account/billing

### EasyOCR lento en primera ejecucion

Es normal: descarga modelos de ~100 MB la primera vez. Las ejecuciones siguientes usan cache.

---

## Comandos rapidos (Windows PowerShell)

```powershell
# Backend
cd agente-sistemas-inteligentes && .\.venv\Scripts\Activate && cd backend && uvicorn app:app --reload --port 8001

# Frontend (otra terminal)
cd agente-sistemas-inteligentes && .\.venv\Scripts\Activate && cd frontend && streamlit run streamlit_app.py
```
