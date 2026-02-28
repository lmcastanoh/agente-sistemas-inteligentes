# RAG Ontologica — Sistema RAG para Fichas Tecnicas Vehiculares

Sistema de Generacion Aumentada por Recuperacion (RAG) especializado en fichas tecnicas
de vehiculos. Permite consultar, resumir y comparar especificaciones de multiples marcas
y modelos usando lenguaje natural.

## Stack Tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend API | FastAPI + Uvicorn |
| Orquestacion | LangGraph (grafo de estados) |
| LLM | OpenAI `gpt-5-nano` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Base vectorial | ChromaDB (persistencia local) |
| Frontend | Streamlit |
| OCR | EasyOCR (paginas escaneadas) |
| Extraccion PDF | pdfplumber |

---

## Estructura del Proyecto

```
rag-ing-ontologica/
│
├── backend/
│   ├── app.py              # API FastAPI: endpoints /ingest y /chat/stream
│   ├── rag_graph.py         # Grafo LangGraph: flujo completo del RAG
│   ├── rag_store.py         # ChromaDB: ingestion, embeddings, vector store
│   ├── tools.py             # Tools de LangGraph (listar, buscar, comparar, resumir)
│   ├── schemas.py           # Modelos Pydantic (IntentClassification, GroundingEvaluation)
│   ├── prompts.py           # System prompts (clasificador, generador, critico)
│   ├── data/                # PDFs organizados por marca (Toyota/, Mazda/, etc.)
│   │   ├── Toyota/
│   │   ├── Mazda/
│   │   ├── Volkswagen/
│   │   ├── Peugeot/
│   │   ├── Opel/
│   │   ├── MG Emotor/
│   │   └── Seat/
│   └── chroma_db/           # Base vectorial persistida (SQLite + HNSW)
│
├── frontend/
│   └── streamlit_app.py     # Interfaz de chat Streamlit
│
├── .env                     # Variables de entorno (OPENAI_API_KEY, HF_TOKEN)
├── requeriments.txt         # Dependencias Python
└── README.md
```

---

## Datos Disponibles

- **6 marcas**: Toyota, Mazda, Volkswagen, Peugeot, Opel, MG Emotor, Seat
- **50 modelos** indexados
- **584 chunks** en ChromaDB
- Metadata por chunk: `source`, `page`, `marca`, `modelo`, `doc_id`, `chunk_id`, `ocr`

---

## Flujo General del Grafo RAG

El sistema usa un grafo de estados LangGraph con 8 nodos y 4 rutas posibles:

```
START
  │
  ▼
┌─────────────────────┐
│  1. classify_intent  │  LLM clasifica la pregunta en 4 intents
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
                       │  3. decide_tools  │  ¿Comparacion o Resumen?
                       └────────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              usar_tools=true         usar_tools=false
                    │                       │
                    ▼                       │
           ┌───────────────┐                │
           │  4. call_tools │  LLM genera   │
           └───────┬───────┘  tool_calls    │
                   │                        │
                   ▼                        │
           ┌───────────────┐                │
           │   5. tools     │  Ejecuta      │
           └───────┬───────┘  las tools     │
                   │                        │
                   ▼                        │
           ┌───────────────────────┐◄───────┘
           │ 6. generate_grounded  │  Genera respuesta con citas
           └───────────┬───────────┘
                       │        ▲
                       ▼        │ retry (score < 0.5 y retry < 1)
           ┌───────────────────────┐
           │ 7. evaluate_grounding │  Critico evalua calidad
           └───────┬───────┬───────┘
                   │       │
              aprobado   rechazado + max retries
                   │       │
                   ▼       ▼
                  END   fallback seguro → END
```

### Rutas del grafo

| Ruta | Intent | Nodos | Descripcion |
|------|--------|-------|-------------|
| **A** | GENERAL | classify → answer_general → END | Conocimiento general sin retrieval |
| **B** | Busqueda | classify → retrieve → decide_tools → generate → evaluate → END | Dato tecnico puntual |
| **C** | Resumen | classify → retrieve → decide_tools → call_tools → tools → generate → evaluate → END | Ficha completa con tool `resumir_ficha` |
| **D** | Comparacion | classify → retrieve → decide_tools → call_tools → tools → generate → evaluate → END | Tabla comparativa con tool `comparar_modelos` |

### Detalle de cada nodo

#### 1. `classify_intent` — Clasificador de intencion
- **LLM**: gpt-5-nano (temperature=0)
- Clasifica en: Busqueda, Resumen, Comparacion, GENERAL
- Extrae entidades: marca, modelo, ano, version
- Sugiere `suggested_k` (cuantos chunks recuperar)
- Memory: si no hay modelo en la pregunta, usa `last_model`/`last_make` del turno anterior
- Keyword fallback: regex corrige clasificaciones erroneas

#### 2. `retrieve` — Recuperacion semantica
- Busca en ChromaDB usando `similarity_search`
- **Dynamic k**: usa `suggested_k` del clasificador o mapa fijo
- Reescribe la query para resolver referencias ("ese modelo" → nombre real)
- Filtros de metadata por marca/modelo con variantes normalizadas
- Comparaciones: retrieval balanceado (k/2 por cada modelo)

#### 3. `decide_tools` — Decision determinista
- Comparacion y Resumen siempre activan tools
- Busqueda no usa tools (genera directo desde contexto)
- Keyword fallback como segunda red de seguridad

#### 4. `call_tools` + `tools` — Ejecucion de herramientas
- **Tools disponibles**:
  - `listar_modelos_disponibles` — catalogo de modelos indexados
  - `buscar_especificacion` — dato tecnico puntual
  - `buscar_por_marca` — todos los modelos de una marca
  - `comparar_modelos` — tabla comparativa markdown
  - `resumir_ficha` — resumen estructurado markdown

#### 5. `generate_grounded` — Generacion con grounding
- Combina contexto de retrieval + output de tools
- Citas obligatorias: `[doc_id=<archivo>; pagina=<N>]`
- Si es reintento, incluye feedback del critico como correccion
- Guarda trazabilidad: prompt completo, snippets de chunks

#### 6. `evaluate_grounding` — Critico de calidad
- Evalua: soportada por contexto, tiene citas, es completa
- Score 0.0 - 1.0
- Si rechazada (score < 0.5) y hay reintentos disponibles → vuelve a generar
- Si rechazada sin reintentos → respuesta fallback segura
- Maximo 1 reintento (configurable con `MAX_RETRIES`)

### Features adicionales

- **Regeneration loop**: el critico puede rechazar y forzar un reintento con feedback correctivo
- **Memory conversacional**: `last_model`/`last_make` persisten entre turnos con reducer `_keep_latest`
- **Trazabilidad completa**: cada nodo registra su paso, decisiones, chunks, prompt enviado
- **Anti-hallucination**: keyword override, grounding estricto, fallback seguro

---

## Configuracion en Windows

### Requisitos previos

- **Python 3.12** — descargar de https://www.python.org/downloads/
- **Git** — descargar de https://git-scm.com/
- Cuenta de OpenAI con API key activa (para gpt-5-nano)

### 1. Clonar el repositorio

```powershell
git clone https://github.com/lmcastanoh/rag-ing-ontologica.git
cd rag-ing-ontologica
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
cd rag-ing-ontologica
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
cd rag-ing-ontologica
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

Chat con streaming via Server-Sent Events (SSE).

```json
// Request
{"question": "¿Cual es la potencia del Toyota Hilux?", "session_id": "sesion-1"}
```

Eventos SSE:
- `token` — respuesta final del RAG
- `trazabilidad` — JSON con la ruta completa, decisiones, chunks, evaluacion
- `done` — fin del stream

---

## Solucion de problemas

### Puerto en uso

```powershell
# Encontrar proceso usando el puerto
netstat -ano | findstr :8001
# Terminar proceso (reemplazar PID)
taskkill /PID <PID> /F
# O usar otro puerto
uvicorn app:app --reload --port 8002
```

### Error de OpenAI 401

Verificar que `OPENAI_API_KEY` esta configurada en `.env`.

### Error de OpenAI 429 (cuota excedida)

Verificar creditos en: https://platform.openai.com/account/billing

### Streamlit no encontrado

```powershell
pip install streamlit
```

### EasyOCR lento en primera ejecucion

Es normal: descarga modelos de ~100 MB la primera vez. Las ejecuciones siguientes usan cache.

---

## Visualizar estructura del grafo

El grafo se imprime en ASCII al iniciar el backend. Para generar diagrama Mermaid:

```python
# En rag_graph.py
print(graph.get_graph().draw_mermaid())
```

Pegar el resultado en: https://mermaid.live

---

## Comandos rapidos (Windows PowerShell)

```powershell
# Backend
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd backend && uvicorn app:app --reload --port 8001

# Frontend (otra terminal)
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd frontend && streamlit run streamlit_app.py
```
