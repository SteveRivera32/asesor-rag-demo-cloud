#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import fitz

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_asesor")
COLLECTION_NAME = "asesor_docs"


def _embedding_fn():
    if embedding_functions is None:
        raise RuntimeError("Falta chromadb. Instala requirements.txt")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",
        model_name="text-embedding-3-small",
    )


def _chunks(text, chunk_size=800, overlap=100):
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        frag = text[start:end]
        if end < len(text):
            cut = max(frag.rfind("\n\n"), frag.rfind(". "), frag.rfind("; "))
            if cut > chunk_size // 2:
                frag = frag[: cut + 1]
                end = start + cut + 1
        frag = frag.strip()
        if frag:
            out.append(frag)
        start = end - overlap if overlap < chunk_size else end
    return out


def _extract_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def _pdfs_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    )


def has_index():
    if chromadb is None or not os.path.isdir(CHROMA_DIR):
        return False
    # Si el directorio existe pero no hay persistencia (por ejemplo, en entornos nuevos),
    # asumimos que no hay índice listo.
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return False
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Importante: no instanciamos embedding_function aquí, porque OpenAIEmbeddingFunction
        # exige OPENAI_API_KEY aun cuando solo queremos validar persistencia.
        client.get_collection(name=COLLECTION_NAME)
        return True
    except Exception:
        return False


def build_index_from_folder(folder):
    if chromadb is None:
        raise RuntimeError("Falta chromadb. Instala requirements.txt")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    pdfs = _pdfs_in_folder(folder)
    if not pdfs:
        return 0

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = _embedding_fn()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

    docs, ids, metas = [], [], []
    for pdf in pdfs:
        txt = _extract_pdf(pdf)
        base = os.path.basename(pdf)
        for i, chunk in enumerate(_chunks(txt)):
            docs.append(chunk)
            ids.append(f"{base}_{i}")
            metas.append({"source": base, "chunk_index": i})

    if docs:
        batch = 100
        for i in range(0, len(docs), batch):
            j = min(i + batch, len(docs))
            collection.add(documents=docs[i:j], ids=ids[i:j], metadatas=metas[i:j])
    return len(docs)


def query_context(question, n_results=8, max_context_chars=7000):
    if chromadb is None or not question.strip():
        return "", []
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn())
    res = collection.query(
        query_texts=[question.strip()],
        n_results=min(max(1, n_results), 20),
        include=["documents", "metadatas"],
    )
    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]

    refs = []
    seen = set()
    for m in metadatas:
        source = (m or {}).get("source", "desconocido")
        idx = (m or {}).get("chunk_index", "?")
        key = (source, idx)
        if key in seen:
            continue
        seen.add(key)
        refs.append(f"{source}#chunk_{idx}")

    selected = []
    total = 0
    for d in documents:
        t = (d or "").strip()
        if not t:
            continue
        if total + len(t) > max_context_chars:
            break
        selected.append(t)
        total += len(t)

    return ("\n\n---\n\n".join(selected) if selected else ""), refs
