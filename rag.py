
from pathlib import Path
import json
import re

import ollama
import fitz
import faiss
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"

# Maximum number of passages used for a document overview.
OVERVIEW_MAX_CHUNKS = 24


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


def read_pdfs(pdf_names=None):
    """Read all PDFs or only the specified PDF filenames."""
    records = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if pdf_names is None:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    else:
        pdf_paths = []

        for name in pdf_names:
            safe_name = Path(name).name
            pdf_path = DATA_DIR / safe_name

            if (
                pdf_path.is_file()
                and pdf_path.suffix.lower() == ".pdf"
            ):
                pdf_paths.append(pdf_path)

    if not pdf_paths:
        raise ValueError(
            "No matching PDF files were found in the data folder."
        )

    for pdf_path in pdf_paths:
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


def build_index(pdf_names=None):
    """Build a FAISS index from all PDFs or selected PDFs."""
    records = read_pdfs(pdf_names=pdf_names)

    if not records:
        raise ValueError(
            "No readable PDF text was found. "
            "Check whether the PDF contains selectable text."
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
    print(
        "Indexed PDFs:",
        sorted(set(record["source"] for record in records))
    )


def load_index_and_records():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Index not found. Upload and process a PDF first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    records = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if not records or index.ntotal != len(records):
        raise ValueError(
            "The index and metadata do not match. "
            "Please process the PDF again."
        )

    return index, records


def search_documents(question, top_k=4):
    """Print the most relevant passages for a question."""
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


def is_overview_question(question):
    """Identify questions asking for a broad document summary."""
    text = question.lower().strip()
    text = re.sub(r"[?!.,]+$", "", text)

    patterns = [
        r"\bwhat is this document about\b",
        r"\bwhat is the document about\b",
        r"\bwhat is this pdf about\b",
        r"\bwhat is the pdf about\b",
        r"\bsummarize this document\b",
        r"\bsummarise this document\b",
        r"\bsummarize the document\b",
        r"\bsummarise the document\b",
        r"\bgive me (?:an )?overview of (?:this )?(?:document|pdf)\b",
        r"\bprovide (?:an )?overview of (?:this )?(?:document|pdf)\b",
        r"\bdescribe the contents of (?:this )?(?:document|pdf)\b",
        r"\bmain topics (?:of|in) (?:this )?(?:document|pdf)\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def select_overview_records(records, max_chunks=OVERVIEW_MAX_CHUNKS):
    """
    Select passages spread across the document.

    If the document is short, use every passage. For a larger document,
    sample evenly across the ordered passages to cover different regions.
    """
    total = len(records)

    if total <= max_chunks:
        return records

    indices = [
        round(i * (total - 1) / (max_chunks - 1))
        for i in range(max_chunks)
    ]

    return [records[i] for i in indices]


def ask_question(question, chat_history=None, top_k=4):
    index, records = load_index_and_records()
    model = SentenceTransformer(MODEL_NAME)

    recent_messages = (chat_history or [])[-6:]

    history_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in recent_messages
        if message.get("role") in {"user", "assistant"}
    )

    overview_question = is_overview_question(question)

    if overview_question:
        # Overview questions need broad document coverage, not just
        # the passages most similar to the word "about".
        selected_records = select_overview_records(records)
        context_parts = [
            (
                f"Source: {item['source']}, Page: {item['page']}\n"
                f"{item['text']}"
            )
            for item in selected_records
        ]

        source_scores = {
            (item["source"], item["page"]): None
            for item in selected_records
        }

        print(
            f"Overview mode: using {len(selected_records)} "
            f"passages from {len(records)} total passages."
        )

    else:
        # For specific questions, resolve follow-up references first.
        rewrite_response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user question as a standalone "
                        "search query for retrieving passages from a PDF. "
                        "Use conversation history only to resolve references "
                        "such as 'it' and 'its'. Do not answer the question. "
                        "Return only the rewritten query."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n"
                        f"{history_text or 'No previous conversation.'}\n\n"
                        f"Latest question: {question}"
                    )
                }
            ]
        )

        search_query = (
            rewrite_response["message"]["content"].strip()
            .strip("\"'")
        )

        if not search_query:
            search_query = question

        print(f"Original question: {question}")
        print(f"Search query: {search_query}")

        query_embedding = model.encode(
            [search_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        scores, indices = index.search(
            query_embedding,
            min(top_k, len(records))
        )

        context_parts = []
        source_scores = {}

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = records[int(idx)]

            context_parts.append(
                f"Source: {item['source']}, "
                f"Page: {item['page']}\n{item['text']}"
            )

            source_key = (item["source"], item["page"])
            source_scores[source_key] = max(
                source_scores.get(source_key, float("-inf")),
                float(score)
            )

    if not context_parts:
        return (
            "I could not find enough information in the uploaded "
            "document to answer that question."
        )

    context = "\n\n".join(context_parts)

    if overview_question:
        instructions = (
            "You are a precise PDF question-answering assistant. "
            "The user is asking for an overview of the document. "
            "Summarize its overall purpose and the main topics, tasks, "
            "or sections represented across the supplied passages. "
            "Do not present one passage as the entire document. "
            "The passages are sampled from across the document, so "
            "be clear about the scope of what they support. "
            "Do not invent a title, purpose, topic, or section that "
            "the passages do not support. If the passages do not provide "
            "enough information for a reliable overview, say so. "
            "Use only the supplied document passages, not outside knowledge."
        )
    else:
        instructions = (
            "You are a precise PDF question-answering assistant. "
            "Answer the current question using only information explicitly "
            "supported by the supplied document passages. Do not use "
            "outside knowledge to fill gaps. If the passages do not answer "
            "the question, say: 'I could not find enough information in "
            "the uploaded document to answer that question.' "
            "Conversation history may help resolve references, but it is "
            "not evidence. Do not invent facts or citations."
        )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": (
                    f"Recent conversation (for reference resolution only):\n"
                    f"{history_text or 'No previous conversation.'}\n\n"
                    f"Document passages:\n{context}\n\n"
                    f"Current question: {question}\n\n"
                    "Answer the current question using the document "
                    "passages above."
                )
            }
        ]
    )

    answer = response["message"]["content"].strip()

    source_lines = []
    for (source, page), score in source_scores.items():
        if score is None:
            source_lines.append(f"- {source}, page {page}")
        else:
            source_lines.append(
                f"- {source}, page {page} "
                f"(similarity: {score:.3f})"
            )

    if source_lines:
        answer += "\n\n**Sources:**\n" + "\n".join(source_lines)

    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Document Assistant"
    )

    parser.add_argument(
        "command",
        choices=["build", "search", "ask"]
    )

    parser.add_argument("--question", type=str)

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
