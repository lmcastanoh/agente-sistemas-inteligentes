"""Genera una imagen PNG del grafo LangGraph."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from rag_graph import build_rag_graph  # noqa: E402

graph = build_rag_graph()
png_bytes = graph.get_graph().draw_mermaid_png()

output = Path(__file__).parent / "grafo.png"
output.write_bytes(png_bytes)
print(f"Grafo guardado en: {output}")
