
from pathlib import Path
import json
import re
import ollama

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"


def chunk_text(text, chunk_size=180, overlap=40):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def read_pdfs():
    records = []

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in DATA_DIR.glob("*.pdf"):
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = re.sub(
                    r"\s+", " ", page.get_text()
                ).strip()

                for chunk in chunk_text(text):
                    records.append({
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_num
                    })

    return records


def build_index():
    records = read_pdfs()

    if not records:
        raise ValueError(
            "No PDF text found. Put a PDF containing "
            "selectable text inside the data folder."
        )

    print(f"Found {len(records)} text chunks.")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        [record["text"] for record in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    METADATA_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Success! Indexed {len(records)} chunks.")


def load_index_and_records():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Index not found. Run: python rag.py build"
        )

    index = faiss.read_index(str(INDEX_PATH))

    records = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if not records:
        raise ValueError(
            "The document index is empty. Run the build command again."
        )

    return index, records


def search_documents(question, top_k=4):
    index, records = load_index_and_records()

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(records))
    )

    found = False

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        found = True
        item = records[int(idx)]

        print(f"\nSource: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Similarity: {score:.3f}")
        print(f"Text: {item['text']}")

    if not found:
        print("No relevant document passages were found.")


def ask_question(question, chat_history=None, top_k=4):
    index, records = load_index_and_records()

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(records))
    )

    context_parts = []
    sources = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = records[int(idx)]

        context_parts.append(
            f"Source: {item['source']}, "
            f"Page: {item['page']}\n{item['text']}"
        )

        sources.append(
            (item["source"], item["page"], float(score))
        )

    if not context_parts:
        return "No relevant document passages were found."

    context = "\n\n".join(context_parts)
        # Prepare recent conversation history for follow-up questions
    history_text = ""

    if chat_history:
        recent_messages = chat_history[-6:]

        history_text = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in recent_messages
            if message.get("role") in {"user", "assistant"}
        )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                     "You are a precise PDF question-answering assistant. "
                    "Answer using only information explicitly supported "
                    "by the supplied document context. "
                    "Do not use outside knowledge to fill gaps. "
                    "If the context does not answer the question, say: "
                    "'I could not find enough information in the uploaded "
                    "documents to answer that question.' "
                    "Answer the exact question asked. "
                    "For a simple definition, use 2 to 4 sentences. "
                    "Avoid repetition, unnecessary lists, and unsupported claims. "
                    "Do not claim that something appears in the document unless "
                    "the supplied context supports that claim."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Recent conversation:\n"
                    f"{history_text or 'No previous conversation.'}\n\n"
                    f"Document context:\n{context}\n\n"
                    f"Current question: {question}\n\n"
                    "Answer the current question using the document context. "
                    "Use the conversation only to understand references "
                    "such as 'it', 'that', or 'explain more'. "
                    "Do not treat previous assistant answers as document "
                    "evidence. If the document context does not support "
                    "the answer, say you could not find enough information."
                )
            }
        ]
    )

    answer = response["message"]["content"]

    # Remove duplicate source/page pairs.
    unique_sources = list(dict.fromkeys(sources))

    source_lines = [
        f"- {source}, page {page} "
        f"(similarity: {score:.3f})"
        for source, page, score in unique_sources
    ]

    return (
        answer
        + "\n\n**Sources:**\n"
        + "\n".join(source_lines)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Document Assistant"
    )

    parser.add_argument(
        "command",
        choices=["build", "search", "ask"]
    )

    parser.add_argument(
        "--question",
        type=str
    )

    args = parser.parse_args()

    if args.command == "build":
        build_index()

    elif args.command == "search":
        if not args.question:
            parser.error("search requires --question")

        search_documents(args.question)

    elif args.command == "ask":
        if not args.question:
            parser.error("ask requires --question")

        print(ask_question(args.question))
