#!/usr/bin/env python3
"""
Minimal RAG: build embeddings -> retrieve -> generate with sources.
- No database. Vectors stored in a small JSON file.
- Works with OpenAI Python SDK >= 1.0.0

Usage:
  python rag_minimal.py --build
  python rag_minimal.py --ask "When should I use RAG instead of fine-tuning?"
"""

import os, json, math, argparse, textwrap
from typing import List, Dict, Tuple
from openai import OpenAI

VECTOR_STORE_PATH = "vector_store.json"
DATA_DIR = "data"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.environ.get("CHAT_MODEL",  "gpt-4o-mini")

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def load_docs() -> List[Tuple[str, str]]:
    rows = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith((".txt", ".md")):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                rows.append((path, f.read()))
    return rows

def build_index(client: OpenAI) -> None:
    docs = load_docs()
    records = []
    for path, text in docs:
        # Keep chunks simple: one chunk per file for this intro
        content = textwrap.shorten(text, width=4000, placeholder=" ...")
        emb = client.embeddings.create(model=EMBED_MODEL, input=content).data[0].embedding
        records.append({"path": path, "text": content, "embedding": emb})
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Indexed {len(records)} documents -> {VECTOR_STORE_PATH}")

def search(client: OpenAI, query: str, k: int = 3) -> List[Dict]:
    if not os.path.exists(VECTOR_STORE_PATH):
        raise SystemExit("Vector store not found. Run with --build first.")
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    qvec = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    scored = []
    for rec in records:
        sim = cosine(qvec, rec["embedding"])
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

def make_prompt(query: str, contexts: List[Dict]) -> List[Dict]:
    sources = "\n\n".join([f"[{i+1}] ({c['path']})\n{c['text']}" for i, c in enumerate(contexts)])
    system = (
        "You are a precise assistant that ONLY answers from the provided sources. "
        "If the answer isn't in the sources, say you don't know. Always include a 'Sources:' list "
        "with the file paths you used."
    )
    user = f"Question:\n{query}\n\nSources:\n{sources}\n\nAnswer with short, factual sentences and cite sources."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def answer(client: OpenAI, query: str) -> str:
    ctx = search(client, query)
    messages = make_prompt(query, ctx)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the tiny embedding index")
    parser.add_argument("--ask", type=str, help="Ask a question with retrieval")
    args = parser.parse_args()

    # Ensure API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY env var to use this script.")

    client = OpenAI()

    if args.build:
        build_index(client)
    if args.ask:
        print(answer(client, args.ask))

    if not args.build and not args.ask:
        parser.print_help()

if __name__ == "__main__":
    main()
