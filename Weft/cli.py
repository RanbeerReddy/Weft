from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from Weft.config.settings import settings
from Weft.core.Extract_data import extract_data_from_zip
from Weft.core.reconstructing_chats import main as reconstruct_main
from Weft.core.retrieval import HybridRetriever, RetrievalPipeline, VectorRetriever
from Weft.storage.create_chunks import build_chunks
from Weft.storage.create_convo_msg import parse_export
from Weft.storage.create_embedding import clear_embeddings, create_embeddings
from Weft.utils.logger import logger

app = typer.Typer(add_completion=False, help="Weft CLI for ingestion, indexing, search, and evaluation.")


@app.command()
def init() -> None:
    """Initialize a local Weft workspace by creating an .env file."""
    env_path = Path(".env")
    if env_path.exists():
        typer.echo(".env already exists; leaving it unchanged.")
        return

    template = """# Database config\nDATABASE_URL=postgresql+psycopg2://weft_user:weft_123@localhost:5432/weft_db\n\n# Model config\nEMBEDDING_MODEL=BAAI/bge-small-en-v1.5\nRERANKER_MODEL=BAAI/bge-reranker-base\n\n# Chunking config\nCHUNK_SIZE=1000\nCHUNK_OVERLAP=150\n\n# Data paths\nRAW_DATA_ZIP=Data/Raw Data/reddyranbeer openAI Data.zip\nEXTRACTED_DATA_DIR=Data/Extracted Data/\nMERGED_CONVERSATIONS_FILE=conversations.json\nVAULT_DIR=vault/conversations\n"""
    env_path.write_text(template, encoding="utf-8")
    typer.echo(f"Created {env_path} with sensible defaults.")


@app.command()
def ingest(chat_export: Path) -> None:
    """Extract a ChatGPT export zip and build the local index artifacts."""
    if not chat_export.exists():
        typer.secho(f"Export archive not found: {chat_export}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"Extracting {chat_export}...")
    extract_data_from_zip(str(chat_export), settings.EXTRACTED_DATA_DIR)

    typer.echo("Reconstructing conversations...")
    reconstruct_main()

    typer.echo("Loading conversations into the database...")
    parse_export(str(Path(settings.MERGED_CONVERSATIONS_FILE)))

    typer.echo("Creating chunks...")
    build_chunks()

    typer.echo("Creating embeddings...")
    create_embeddings()


@app.command()
def embed(clear: bool = typer.Option(False, "--clear", help="Clear existing embeddings before generating new ones.")) -> None:
    """Generate embeddings for the existing chunk corpus."""
    if clear:
        clear_embeddings()
    create_embeddings()


@app.command()
def search(query: str, top_k: int = typer.Option(5, "--top-k", help="Number of results to display.")) -> None:
    """Search indexed conversations with the semantic retrieval pipeline."""
    retriever = RetrievalPipeline()
    results = retriever.search(query, top_n=max(top_k * 2, 20), final_k=top_k)
    if not results:
        typer.echo("No results found.")
        return

    typer.echo(f"Found {len(results)} result(s):")
    for item in results:
        typer.echo(
            f"- [{item.rank}] {item.conversation_title or 'Untitled'} | {item.message_role or 'unknown'} | {item.chunk_text[:180]}"
        )


@app.command()
def benchmark() -> None:
    """Print the latest benchmark summary from the bundled benchmark report."""
    benchmark_path = Path("hybrid_benchmark_results.json")
    if not benchmark_path.exists():
        typer.secho("No benchmark report found at hybrid_benchmark_results.json", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    with benchmark_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for name, summary in data.items():
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("mrr")
        if isinstance(metrics, (int, float)):
            typer.echo(f"{name}: MRR={metrics:.3f} | hit@10={summary.get('hit_at_10', 0):.3f}")


@app.command()
def evaluate() -> None:
    """Run the bundled evaluation pipeline summary."""
    from Weft.evaluation.run_all_phases import run_all

    report = run_all()
    typer.echo(f"Evaluation completed with {len(report)} phases.")


@app.command()
def stats() -> None:
    """Print a lightweight inventory of stored conversations, messages, chunks, and embeddings."""
    from Weft.storage.database import SessionLocal
    from Weft.storage.models import Chunk, Conversation, Embedding, Message
    from sqlalchemy import func, select

    db = SessionLocal()
    try:
        counts = {
            "conversations": db.scalar(select(func.count(Conversation.id))),
            "messages": db.scalar(select(func.count(Message.id))),
            "chunks": db.scalar(select(func.count(Chunk.id))),
            "embeddings": db.scalar(select(func.count(Embedding.id))),
        }
    finally:
        db.close()

    typer.echo(json.dumps(counts, indent=2))


if __name__ == "__main__":
    app()
