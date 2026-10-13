r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import warnings
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

TEMPERATURE = 0.2
TOP_K = 5
CANDIDATE_POOL = 15
ADVANCED_PER_QUERY_K = 10
GRAPH_CONTEXT_SLOTS = 2

# Exact Checkpoint 2.1 baseline weights.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

# Exact Checkpoint 2.1 chunking settings.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500

TOPICS_PER_CHUNK = 4
MIN_TOPIC_DF = 3
MAX_TOPIC_DF_RATIO = 0.20

GRAPH_SEED_K = 5
GRAPH_CONTEXT_SLOTS = 2
GRAPH_DIRECT_RERANK_K = 50

SCENARIO = "research_papers"

SCRIPT_DIR = Path(__file__).resolve().parent
COURSE_ROOT = SCRIPT_DIR.parent

LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"
CORPUS_CACHE_PATH = SCRIPT_DIR / "checkpoint_4_1_chunks.jsonl"
DECOMPOSITION_CACHE_PATH = SCRIPT_DIR / "checkpoint_4_1_decomposition_cache.json"
LOCAL_CHROMA_DIR = SCRIPT_DIR / "checkpoint_4_1_chroma"
TOPIC_CACHE_PATH = SCRIPT_DIR / "checkpoint_4_1_topic_cache.json"

# Reuse an existing vector DB before building another one.
CHROMA_CANDIDATES = [
    COURSE_ROOT / "checkpoint_2" / "checkpoint_2_1_chroma",
    COURSE_ROOT / "checkpoint_3" / "checkpoint_3_1_chroma",
    SCRIPT_DIR / "checkpoint_2_1_chroma",
    SCRIPT_DIR / "checkpoint_3_1_chroma",
    LOCAL_CHROMA_DIR,
]

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing. PRIMARY documents were retrieved "
    "directly from the user's original question. CONTEXT documents were surfaced "
    "through query decomposition. Use either when relevant, but base every claim "
    "on the supplied evidence."
)

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a research-paper retrieval system. "
    "Break the user's question into 2-4 focused retrieval sub-queries that together "
    "cover the information needed to answer it. Preserve important technical terms "
    "and named concepts, but let different sub-queries target distinct aspects. "
    "Do not answer the question. Return ONLY a JSON array of strings, for example "
    '["sub-query 1", "sub-query 2"].'
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)

RELEVANCE_SYSTEM = (
    "You are evaluating retrieval quality. "
    "Given a user QUESTION and numbered retrieved CHUNKS, determine whether each "
    "chunk contains information that is useful for answering the question. "
    "Return ONLY a JSON array of true/false values in the same order as the chunks."
)

GROUNDEDNESS_SYSTEM = (
    "You are evaluating groundedness. Given an ANSWER and the retrieved DOCUMENTS "
    "used to produce it, determine whether the substantive claims in the answer are "
    "supported by those documents. Reply with exactly one word: 'pass' if the answer "
    "is adequately supported, otherwise 'fail'."
)

# Additional stopwords used only for automatically derived graph topics.
# These remove generic academic words that are poor relationship labels.
TOPIC_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "among", "another",
    "because", "been", "before", "being", "between", "both", "could", "does",
    "each", "from", "have", "into", "more", "most", "other", "paper", "papers",
    "results", "result", "show", "shows", "some", "such", "than", "their",
    "there", "these", "they", "through", "using", "used", "uses", "very",
    "which", "while", "with", "within", "would", "method", "methods", "approach",
    "approaches", "system", "systems", "model", "models", "data", "problem",
    "problems", "work", "based", "however", "different", "new", "can", "may",
}

# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# Six short "documents" with relationships so both techniques are demonstrable
# without indexing your full corpus. Each has an ID, text, a `topics` list, and
# `links` to related documents (i.e., think citations for papers, or hyperlinks for
# Wikipedia). Your real system would derive these from your actual data.

# %%
def _nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


# locate an existing persisted vector DB.
def find_existing_chroma_dir() -> Path | None:
    override = os.getenv("CAPSTONE_CHROMA_DIR")

    if override:
        override_path = Path(override).expanduser().resolve()
        if _nonempty_dir(override_path):
            return override_path

    for path in CHROMA_CANDIDATES:
        if _nonempty_dir(path):
            return path

    return None

def open_chroma(path: Path) -> Chroma:
    return Chroma(
        persist_directory=str(path),
        embedding_function=get_embeddings(),
    )

def docs_from_chroma(db: Chroma) -> list[Document]:
    """Recover LangChain Documents from the persisted Checkpoint 2 vector store."""

    payload = db.get(include=["documents", "metadatas"])

    ids = payload.get("ids") or []
    texts = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    if not ids or len(ids) != len(texts):
        raise RuntimeError("Persisted Chroma DB did not return a usable document corpus.")

    docs: list[Document] = []

    for index, doc_id in enumerate(ids):
        text = texts[index] or ""
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        metadata["id"] = str(doc_id)

        if "source" not in metadata:
            metadata["source"] = str(doc_id).split("::chunk_")[0]

        if "chunk" not in metadata:
            match = re.search(r"::chunk_(\d+)$", str(doc_id))
            metadata["chunk"] = int(match.group(1)) if match else index

        docs.append(Document(page_content=text, metadata=metadata))

    if not docs:
        raise RuntimeError("No documents were recovered from the persisted Chroma DB.")

    return docs

def save_chunk_cache(docs: list[Document]) -> None:
    with CORPUS_CACHE_PATH.open("w", encoding="utf-8") as fh:
        for doc in docs:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunk_cache() -> list[Document]:
    docs: list[Document] = []

    with CORPUS_CACHE_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            docs.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"],
                )
            )

    if not docs:
        raise RuntimeError("Chunk cache exists but contains no documents.")

    return docs


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_point = text.rfind(" ", start, end)
            if split_point > start + chunk_size // 2:
                end = split_point

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def load_research_papers(corpus_dir: Path) -> list[Document]:
    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {corpus_dir}")

    docs: list[Document] = []

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        paper_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(paper_text)

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.name}::chunk_{chunk_number:04d}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "id": chunk_id,
                        "source": pdf_path.name,
                        "chunk": chunk_number,
                    },
                )
            )

    print(f"Loaded {len(pdf_files)} papers into {len(docs)} chunks from PDFs.")
    return docs


def resolve_corpus_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    env_path = os.getenv("CAPSTONE_RESEARCH_PAPERS")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        COURSE_ROOT / "ResearchPapers",
        COURSE_ROOT / "Labs" / "CapstoneDatasets" / "ResearchPapers",
        SCRIPT_DIR / "ResearchPapers",
    ]

    for path in candidates:
        if path.is_dir():
            return path

    return COURSE_ROOT / "ResearchPapers"


def build_local_db(docs: list[Document]) -> Chroma:
    print("Building local Checkpoint 4 vector DB — embedding research paper chunks...")

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=str(LOCAL_CHROMA_DIR),
        ids=[str(d.metadata["id"]) for d in docs],
    )


def load_corpus_and_db() -> tuple[list[Document], Chroma]:
    """Load existing data first; parse PDFs only if every reusable source fails."""

    existing_chroma = find_existing_chroma_dir()

    if existing_chroma is not None:
        try:
            print(f"Reusing vector DB from {existing_chroma}")
            db = open_chroma(existing_chroma)
            docs = docs_from_chroma(db)
            print(
                f"Recovered {len(docs)} chunks directly from Chroma; "
                "skipped PDF extraction and re-embedding."
            )
            return docs, db
        except Exception as exc:
            print(
                "Existing Chroma DB could not be used as the corpus source "
                f"({exc}). Falling back to cached/extracted chunks."
            )

    if CORPUS_CACHE_PATH.is_file():
        try:
            docs = load_chunk_cache()
            print(
                f"Loaded {len(docs)} chunks from {CORPUS_CACHE_PATH.name}; "
                "skipped PDF extraction."
            )

            if _nonempty_dir(LOCAL_CHROMA_DIR):
                print(f"Loading local vector DB from {LOCAL_CHROMA_DIR}")
                return docs, open_chroma(LOCAL_CHROMA_DIR)

            return docs, build_local_db(docs)
        except Exception as exc:
            print(f"Chunk cache could not be reused ({exc}). Falling back to PDFs.")

    corpus_dir = resolve_corpus_dir()

    if not corpus_dir.is_dir():
        raise RuntimeError(
            "No reusable Chroma DB or chunk cache was found, and the ResearchPapers "
            f"directory does not exist at: {corpus_dir}\n"
            "Pass the ResearchPapers path as the first command-line argument or set "
            "CAPSTONE_RESEARCH_PAPERS."
        )

    docs = load_research_papers(corpus_dir)
    save_chunk_cache(docs)
    print(f"Saved reusable chunk cache to {CORPUS_CACHE_PATH}")

    if _nonempty_dir(LOCAL_CHROMA_DIR):
        print(f"Loading local vector DB from {LOCAL_CHROMA_DIR}")
        return docs, open_chroma(LOCAL_CHROMA_DIR)

    return docs, build_local_db(docs)


# %% [markdown]
# ## Checkpoint 2.1 hybrid baseline retriever — unchanged

# %%
STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be",
    "this", "that", "these", "those",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    ]


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if high == low:
        return [0.5] * len(scores)

    normalized = [(score - low) / (high - low) for score in scores]

    if invert:
        normalized = [1.0 - score for score in normalized]

    return normalized


class HybridRetriever:
    """Equal-weight BM25 + Chroma vector retrieval from Checkpoint 2.1."""

    def __init__(self, docs: list[Document], db: Chroma):
        self.docs = docs
        self.db = db
        self.doc_by_id = {
            str(doc.metadata["id"]): doc
            for doc in docs
        }
        self.ids = [str(doc.metadata["id"]) for doc in docs]
        self.bm25 = BM25Okapi([_tokens(doc.page_content) for doc in docs])

    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokens(query))
        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        return [(self.ids[i], float(scores[i])) for i in indices]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self.db.similarity_search_with_score(query, k=k)
        return [
            (str(doc.metadata["id"]), float(distance))
            for doc, distance in results
        ]

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
    ) -> list[tuple[str, float]]:
        bm25_results = self._bm25_topk(query, CANDIDATE_POOL)
        vector_results = self._vector_topk(query, CANDIDATE_POOL)

        bm25_scores = _normalize([score for _, score in bm25_results])
        vector_scores = _normalize(
            [score for _, score in vector_results],
            invert=True,
        )

        candidates: dict[str, dict[str, float]] = {}

        for (doc_id, _), score in zip(bm25_results, bm25_scores):
            candidates.setdefault(doc_id, {"bm25": 0.0, "vector": 0.0})
            candidates[doc_id]["bm25"] = score

        for (doc_id, _), score in zip(vector_results, vector_scores):
            candidates.setdefault(doc_id, {"bm25": 0.0, "vector": 0.0})
            candidates[doc_id]["vector"] = score

        fused: list[tuple[str, float]] = []

        for doc_id, scores in candidates.items():
            fused_score = (
                WEIGHT_BM25 * scores["bm25"]
                + WEIGHT_VECTOR * scores["vector"]
            )
            fused.append((doc_id, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(
    llm: ChatOpenAI,
    retriever: HybridRetriever,
    query: str,
    k: int = TOP_K,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Original query + decomposed sub-queries, all using the same hybrid retriever."""

    sub_queries = decompose_query(llm, query)

    baseline_top = retriever.retrieve(query, k)
    baseline_top_ids = {
        doc_id
        for doc_id, _ in baseline_top
    }

    arms: list[tuple[str, str]] = [
        ("original", query)
    ]

    for i, sub_query in enumerate(sub_queries, 1):
        if sub_query.strip().lower() == query.strip().lower():
            continue

        arms.append(
            (f"subquery_{i}", sub_query)
        )

    score_map: dict[str, float] = {}
    matched_by: dict[str, list[str]] = {}

    for arm_name, arm_query in arms:

        hits = retriever.retrieve(
            arm_query,
            ADVANCED_PER_QUERY_K,
        )

        for doc_id, score in hits:

            score_map[doc_id] = (
                score_map.get(doc_id, 0.0)
                + score
            )

            matched_by.setdefault(
                doc_id,
                [],
            ).append(arm_name)

    ranked = sorted(
        score_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    results: list[dict[str, Any]] = []

    for doc_id, aggregate_score in ranked:

        results.append(
            {
                "doc_id": doc_id,
                "score": aggregate_score,
                "role": (
                    "primary"
                    if doc_id in baseline_top_ids
                    else "context"
                ),
                "matched_by": matched_by.get(
                    doc_id,
                    [],
                ),
            }
        )

    return results, sub_queries


# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)
#
# Build a graph whose nodes are documents and topics, with edges for "links to" and
# "relates to topic." Retrieval seeds with the baseline hits, then pulls in each
# seed's linked documents and topic-siblings as extra context — deduplicated.

# %%
# stable corpus signature for graph-topic cache validation.
def _corpus_signature(docs: list[Document]) -> str:
    joined = "\n".join(sorted(str(doc.metadata["id"]) for doc in docs))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# tokenize text specifically for graph-topic extraction.
def _topic_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
        if token not in STOPWORDS and token not in TOPIC_STOPWORDS
    ]


# derive TF-IDF-like topic terms for each chunk.
def _derive_topics_by_chunk(docs: list[Document]) -> dict[str, list[str]]:
    token_counters: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()

    for doc in docs:
        doc_id = str(doc.metadata["id"])
        counts = Counter(_topic_tokens(doc.page_content))
        token_counters[doc_id] = counts
        document_frequency.update(counts.keys())

    n_docs = max(len(docs), 1)
    max_df = max(MIN_TOPIC_DF, int(n_docs * MAX_TOPIC_DF_RATIO))
    topics_by_chunk: dict[str, list[str]] = {}

    for doc_id, counts in token_counters.items():
        weighted: list[tuple[str, float]] = []

        for term, tf in counts.items():
            df = document_frequency[term]

            if df < MIN_TOPIC_DF or df > max_df:
                continue

            idf = math.log((n_docs + 1) / (df + 1)) + 1.0
            weighted.append((term, float(tf) * idf))

        weighted.sort(key=lambda item: item[1], reverse=True)
        topics_by_chunk[doc_id] = [term for term, _ in weighted[:TOPICS_PER_CHUNK]]

    return topics_by_chunk


# save automatically derived graph topics.
def _save_topic_cache(docs: list[Document], topics_by_chunk: dict[str, list[str]]) -> None:
    payload = {
        "corpus_signature": _corpus_signature(docs),
        "document_count": len(docs),
        "topics_per_chunk": TOPICS_PER_CHUNK,
        "topics_by_chunk": topics_by_chunk,
    }
    TOPIC_CACHE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


# load graph topics when the corpus signature matches.
def _load_topic_cache(docs: list[Document]) -> dict[str, list[str]] | None:
    if not TOPIC_CACHE_PATH.is_file():
        return None

    try:
        payload = json.loads(TOPIC_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if payload.get("corpus_signature") != _corpus_signature(docs):
        return None

    raw_topics = payload.get("topics_by_chunk")
    if not isinstance(raw_topics, dict):
        return None

    return {
        str(doc_id): [str(topic) for topic in topics]
        for doc_id, topics in raw_topics.items()
        if isinstance(topics, list)
    }


# build_graph() adapted from the sample document/topic graph to the real corpus.
def build_graph(docs: list[Document]) -> nx.DiGraph:
    topics_by_chunk = _load_topic_cache(docs)

    if topics_by_chunk is None:
        print("Deriving topic relationships for graph retrieval...")
        topics_by_chunk = _derive_topics_by_chunk(docs)
        _save_topic_cache(docs, topics_by_chunk)
        print(f"Saved graph-topic cache to {TOPIC_CACHE_PATH}")
    else:
        print(f"Reusing graph-topic cache from {TOPIC_CACHE_PATH}")

    graph = nx.DiGraph()

    for doc in docs:
        doc_id = str(doc.metadata["id"])
        source = str(doc.metadata["source"])
        chunk_number = int(doc.metadata["chunk"])

        paper_node = f"paper:{source}"
        chunk_node = f"chunk:{doc_id}"

        graph.add_node(
            paper_node,
            node_type="paper",
            source=source,
        )
        graph.add_node(
            chunk_node,
            node_type="chunk",
            doc_id=doc_id,
            source=source,
            chunk=chunk_number,
        )
        graph.add_edge(
            paper_node,
            chunk_node,
            edge_type="contains",
        )

        for topic in topics_by_chunk.get(doc_id, []):
            topic_node = f"topic:{topic}"
            graph.add_node(
                topic_node,
                node_type="topic",
                topic=topic,
            )
            graph.add_edge(
                chunk_node,
                topic_node,
                edge_type="relates_to",
            )

    return graph


# normalize a score mapping for graph reranking.
def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    low = min(values)
    high = max(values)

    if high == low:
        return {key: 0.5 for key in scores}

    return {
        key: (value - low) / (high - low)
        for key, value in scores.items()
    }


# graph_retrieve() adapted from the starter's topic-neighbor expansion.
def graph_retrieve(
    graph: nx.DiGraph,
    query: str,
    seed_hits: list[dict[str, Any]],
    retriever: HybridRetriever,
    context_k: int = GRAPH_CONTEXT_SLOTS,
) -> list[dict[str, Any]]:
    """Expand seed chunks through shared-topic edges to related chunks in other papers."""

    seed_ids = [str(hit["doc_id"]) for hit in seed_hits[:GRAPH_SEED_K]]
    seed_sources = {
        str(retriever.doc_by_id[doc_id].metadata["source"])
        for doc_id in seed_ids
        if doc_id in retriever.doc_by_id
    }

    graph_affinity: dict[str, float] = defaultdict(float)
    shared_topics: dict[str, set[str]] = defaultdict(set)

    for seed_id in seed_ids:
        seed_node = f"chunk:{seed_id}"

        if not graph.has_node(seed_node):
            continue

        for _, topic_node, edge_data in graph.out_edges(seed_node, data=True):
            if edge_data.get("edge_type") != "relates_to":
                continue

            topic_name = str(graph.nodes[topic_node].get("topic", topic_node))
            siblings = list(graph.in_edges(topic_node, data=True))
            sibling_count = max(len(siblings), 1)

            # Rarer shared topics provide a stronger relationship signal.
            topic_weight = 1.0 / math.log2(sibling_count + 1.0)

            for sibling_node, _, sibling_edge in siblings:
                if sibling_edge.get("edge_type") != "relates_to":
                    continue

                candidate_id = str(graph.nodes[sibling_node].get("doc_id", ""))

                if not candidate_id or candidate_id in seed_ids:
                    continue
                if candidate_id not in retriever.doc_by_id:
                    continue

                candidate_source = str(
                    retriever.doc_by_id[candidate_id].metadata["source"]
                )

                # Graph expansion is intended to add cross-paper context.
                if candidate_source in seed_sources:
                    continue

                graph_affinity[candidate_id] += topic_weight
                shared_topics[candidate_id].add(topic_name)

    if not graph_affinity:
        return []

    # Add a small direct-query relevance signal so graph neighbors are still useful
    # for the user's actual question rather than being related only by topic.
    direct_scores = dict(retriever.retrieve(query, GRAPH_DIRECT_RERANK_K))
    graph_scores_norm = _normalize_score_map(dict(graph_affinity))

    candidate_scores: list[tuple[str, float]] = []

    for doc_id, graph_score in graph_scores_norm.items():
        direct_score = direct_scores.get(doc_id, 0.0)
        combined_score = 0.70 * graph_score + 0.30 * direct_score
        candidate_scores.append((doc_id, combined_score))

    candidate_scores.sort(key=lambda item: item[1], reverse=True)

    results: list[dict[str, Any]] = []

    for doc_id, score in candidate_scores[:context_k]:
        results.append(
            {
                "doc_id": doc_id,
                "score": score,
                "role": "context",
                "matched_by": ["graph"],
                "shared_topics": sorted(shared_topics.get(doc_id, set())),
            }
        )

    return results


# combine decomposition seeds with graph context at matched TOP_K.
def advanced_retrieve(
    llm: ChatOpenAI,
    graph: nx.DiGraph,
    retriever: HybridRetriever,
    query: str,
    k: int = TOP_K,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Final advanced strategy: decomposition first, then graph context expansion."""

    decomposition_hits, sub_queries = multistep_retrieve(
        llm,
        retriever,
        query,
        k,
    )

    graph_hits = graph_retrieve(
        graph,
        query,
        decomposition_hits,
        retriever,
        context_k=min(GRAPH_CONTEXT_SLOTS, k),
    )

    graph_ids = {str(hit["doc_id"]) for hit in graph_hits}
    primary_slots = max(k - len(graph_hits), 0)

    final_hits: list[dict[str, Any]] = []

    for hit in decomposition_hits:
        if len(final_hits) >= primary_slots:
            break
        if str(hit["doc_id"]) in graph_ids:
            continue

        final_hits.append(
            {
                **hit,
                "role": "primary",
                "shared_topics": [],
            }
        )

    final_hits.extend(graph_hits)

    # If graph expansion produced fewer context docs than requested, fill from the
    # remaining decomposition results so baseline and advanced retain the same TOP_K.
    selected_ids = {str(hit["doc_id"]) for hit in final_hits}

    for hit in decomposition_hits:
        if len(final_hits) >= k:
            break
        if str(hit["doc_id"]) in selected_ids:
            continue

        final_hits.append(
            {
                **hit,
                "role": "primary",
                "shared_topics": [],
            }
        )
        selected_ids.add(str(hit["doc_id"]))

    return final_hits[:k], sub_queries, decomposition_hits

# ## Grounded answer generation and evaluation

# baseline answer helper using real chunk Documents.
def answer_from_baseline(
    llm: ChatOpenAI,
    query: str,
    hits: list[tuple[str, float]],
    doc_by_id: dict[str, Document],
) -> str:
    context_parts: list[str] = []

    for doc_id, _ in hits:
        doc = doc_by_id[doc_id]
        context_parts.append(
            f"[PRIMARY | Source: {doc.metadata['source']} | "
            f"Chunk: {doc.metadata['chunk']}]\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]

    return llm.invoke(messages).content


# answer_from_docs() adapted to PRIMARY/CONTEXT real-corpus evidence.
def answer_from_docs(
    llm: ChatOpenAI,
    query: str,
    hits: list[dict[str, Any]],
    doc_by_id: dict[str, Document],
) -> str:
    context_parts: list[str] = []

    for hit in hits:
        doc = doc_by_id[str(hit["doc_id"])]
        role = str(hit.get("role", "context")).upper()
        matched = ", ".join(hit.get("matched_by", [])) or "unknown"
        topics = ", ".join(hit.get("shared_topics", []))
        graph_note = f" | Shared graph topics: {topics}" if topics else ""

        context_parts.append(
            f"[{role} | Source: {doc.metadata['source']} | "
            f"Chunk: {doc.metadata['chunk']} | Retrieved by: {matched}{graph_note}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]

    return llm.invoke(messages).content


def judge(
    llm: ChatOpenAI,
    answer_text: str,
    grading_notes: str,
) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=(
                f"ANSWER:\n{answer_text}\n\n"
                f"GRADING NOTES:\n{grading_notes}\n\n"
                "Verdict (pass/fail):"
            )
        ),
    ]

    verdict = llm.invoke(messages).content.strip().lower()
    print(f"CORRECTNESS RAW JUDGE: {verdict}")
    return "pass" if "pass" in verdict else "fail"


# same evaluation set for an apples-to-apples comparison.
def my_eval_set() -> list[dict[str, str]]:
    return [
        {
            "question":
                "How does the sketching approach to program synthesis work?",
            "grading_notes":
                "Explains that sketching uses a partial program that specifies "
                "the high-level structure, while an automated synthesis process "
                "fills in missing details subject to a specification.",
        },
        {
            "question":
                "How do program synthesis systems use specifications or examples "
                "to generate programs?",
            "grading_notes":
                "Explains that specifications such as input-output examples are "
                "used as constraints that guide the search for a program satisfying "
                "the required behavior.",
        },
        {
            "question":
                "What limitations or open problems do the papers identify for "
                "program synthesis?",
            "grading_notes":
                "Identifies at least two supported limitations, such as computational "
                "difficulty or large search spaces and usability or tight coupling "
                "between specifications and synthesis solvers.",
        },
        {
            "question":
                "How can a partially written program have its missing implementation "
                "details automatically completed?",
            "grading_notes":
                "Recognizes the idea as program sketching or synthesis and explains "
                "that an automated synthesizer fills in unspecified implementation "
                "details while satisfying a specification or correctness criteria.",
        },
        {
            "question":
                "Compare the major approaches to program synthesis discussed across "
                "the papers.",
            "grading_notes":
                "Correctly distinguishes at least three supported synthesis approaches, "
                "such as syntax-guided synthesis, SMT or refutation-based synthesis, "
                "adaptive concretization, or programmable synthesis.",
        },
    ]

# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%
def my_advanced_plan() -> dict[str, Any]:
    """Return YOUR advanced-retrieval plan for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "technique":   "decomposition", "graph", or "both"
      - "node_types":  list[str]  — for graph: the kinds of nodes in your corpus
                       (e.g., ["paper", "author", "topic"] or ["article", "category"])
      - "edge_types":  list[str]  — the relationships you will surface
                       (e.g., ["cites", "same_author", "shares_topic"])
      - "test_queries": list[str] — 2-3 questions from your 3.1 evaluation set that
                       single-pass retrieval handled poorly and that should improve.
      - "rationale":   str — one or two sentences on why these choices fit your data.

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "technique": "both",
        "node_types": ["paper", "chunk", "topic"],
        "edge_types": ["contains", "relates_to"],
        "test_queries": [
            "How can a partially written program have its missing implementation "
            "details automatically completed?",
            "Compare the major approaches to program synthesis discussed across "
            "the papers.",
            "What limitations or open problems do the papers identify for "
            "program synthesis?",
        ],
        "rationale": (
            "Checkpoint 3.1 showed that final answers could still pass even when "
            "retrieval ranking was imperfect, especially for paraphrased and broad "
            "multi-paper questions. Query decomposition searches each aspect "
            "separately, while graph expansion adds related chunks from other papers "
            "through shared topic relationships that a single similarity search may miss."
        ),
    }


# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass retrieval with the two advanced retrieval strategies 
# using the sample corpus and a representative query. It also logs the result. Read the
# retrieved sets: The advanced strategies should surface related documents that the
# baseline misses. Then run the same comparison in your real system for your report.

# %%
# source-diversity metric helper.
def _baseline_sources(
    hits: list[tuple[str, float]],
    doc_by_id: dict[str, Document],
) -> set[str]:
    return {
        str(doc_by_id[doc_id].metadata["source"])
        for doc_id, _ in hits
    }


# source-diversity metric helper.
def _advanced_sources(
    hits: list[dict[str, Any]],
    doc_by_id: dict[str, Document],
) -> set[str]:
    return {
        str(doc_by_id[str(hit["doc_id"])].metadata["source"])
        for hit in hits
    }


# compare retrieved-set overlap.
def _retrieval_overlap(
    baseline_hits: list[tuple[str, float]],
    advanced_hits: list[dict[str, Any]],
) -> tuple[int, float]:
    baseline_ids = {doc_id for doc_id, _ in baseline_hits}
    advanced_ids = {str(hit["doc_id"]) for hit in advanced_hits}

    intersection = len(baseline_ids & advanced_ids)
    union = len(baseline_ids | advanced_ids)
    jaccard = intersection / union if union else 1.0

    return intersection, jaccard

# convert baseline or advanced hits to Documents.
def _docs_for_evaluation(
    hits,
    doc_by_id: dict[str, Document],
) -> list[Document]:

    docs: list[Document] = []

    for hit in hits:
        if isinstance(hit, tuple):
            # Baseline format: (doc_id, score)
            doc_id = str(hit[0])
        else:
            # Advanced format: {"doc_id": ..., ...}
            doc_id = str(hit["doc_id"])

        docs.append(doc_by_id[doc_id])

    return docs

# same retrieval relevance metric for both systems.
def retrieval_relevance(
    llm: ChatOpenAI,
    query: str,
    docs: list[Document],
) -> tuple[int, int, float]:

    if not docs:
        return 0, 0, 0.0

    chunks = "\n\n".join(
        f"[{i}] {doc.page_content[:2500]}"
        for i, doc in enumerate(docs, 1)
    )

    messages = [
        SystemMessage(content=RELEVANCE_SYSTEM),
        HumanMessage(
            content=(
                f"QUESTION:\n{query}\n\n"
                f"CHUNKS:\n{chunks}\n\n"
                "Relevant chunks:"
            )
        ),
    ]

    raw = llm.invoke(messages).content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if "[" in raw and "]" in raw:
            raw = raw[raw.find("["): raw.rfind("]") + 1]

    try:
        values = json.loads(raw)

        if (
            isinstance(values, list)
            and len(values) == len(docs)
            and all(isinstance(x, bool) for x in values)
        ):
            relevant = sum(values)
            total = len(values)
            return relevant, total, relevant / total

    except (json.JSONDecodeError, ValueError):
        pass

    return 0, len(docs), 0.0

# same groundedness metric for both systems.
def groundedness(
    llm: ChatOpenAI,
    answer_text: str,
    docs: list[Document],
) -> str:

    if not docs:
        return "fail"

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
    )

    messages = [
        SystemMessage(content=GROUNDEDNESS_SYSTEM),
        HumanMessage(
            content=(
                f"DOCUMENTS:\n{context}\n\n"
                f"ANSWER:\n{answer_text}\n\n"
                "Verdict (pass/fail):"
            )
        ),
    ]

    verdict = llm.invoke(messages).content.strip().lower()

    print(f"GROUNDEDNESS RAW JUDGE: {verdict}")

    if verdict == "pass":
        return "pass"

    return "fail"

# real-system baseline vs decomposition+graph evaluation.
def run_comparison(
    llm: ChatOpenAI,
    graph: nx.DiGraph,
    retriever: HybridRetriever,
) -> None:
    eval_set = my_eval_set()

    # Existing answer-correctness and retrieval-diversity totals.
    baseline_passes = 0
    advanced_passes = 0
    baseline_source_total = 0
    advanced_source_total = 0
    graph_context_total = 0

    # ADDED FOR CAPSTONE EVALUATION — shared metrics for both systems.
    baseline_relevance_total = 0.0
    advanced_relevance_total = 0.0

    baseline_grounded_passes = 0
    advanced_grounded_passes = 0

    print(
        f"\nCheckpoint 4.1 — baseline vs advanced retrieval"
        f" | scenario: {SCENARIO}\n"
    )
    print(
        f"Matched context count: baseline TOP_K={TOP_K}, "
        f"advanced TOP_K={TOP_K}\n"
    )
    print(
        "Advanced strategy: query decomposition + existing hybrid retrieval "
        "+ shared-topic graph expansion\n"
    )

    for i, item in enumerate(eval_set, 1):
        query = item["question"]
        grading_notes = item["grading_notes"]

        # ------------------------------------------------------------
        # Retrieve baseline and advanced evidence.
        # ------------------------------------------------------------

        baseline_hits = retriever.retrieve(query, TOP_K)

        advanced_hits, sub_queries, decomposition_hits = advanced_retrieve(
            llm,
            graph,
            retriever,
            query,
            TOP_K,
        )

        # ------------------------------------------------------------
        # Generate answers.
        # ------------------------------------------------------------

        baseline_answer = (
            answer_from_baseline(
                llm,
                query,
                baseline_hits,
                retriever.doc_by_id,
            )
            if baseline_hits
            else "(no documents retrieved)"
        )

        advanced_answer = (
            answer_from_docs(
                llm,
                query,
                advanced_hits,
                retriever.doc_by_id,
            )
            if advanced_hits
            else "(no documents retrieved)"
        )

        # ------------------------------------------------------------
        # Metric 1 — answer correctness.
        # Same Checkpoint 3.1 judge for both systems.
        # ------------------------------------------------------------

        baseline_verdict = judge(
            llm,
            baseline_answer,
            grading_notes,
        )

        advanced_verdict = judge(
            llm,
            advanced_answer,
            grading_notes,
        )

        # ------------------------------------------------------------
        # Convert both retrieval outputs into the same Document format
        # so the new metrics are applied identically.
        # ------------------------------------------------------------

        baseline_eval_docs = _docs_for_evaluation(
            baseline_hits,
            retriever.doc_by_id,
        )

        advanced_eval_docs = _docs_for_evaluation(
            advanced_hits,
            retriever.doc_by_id,
        )

        # ------------------------------------------------------------
        # Metric 2 — retrieval relevance@K.
        # ------------------------------------------------------------

        (
            baseline_relevant,
            baseline_retrieved,
            baseline_relevance,
        ) = retrieval_relevance(
            llm,
            query,
            baseline_eval_docs,
        )

        (
            advanced_relevant,
            advanced_retrieved,
            advanced_relevance,
        ) = retrieval_relevance(
            llm,
            query,
            advanced_eval_docs,
        )

        # ------------------------------------------------------------
        # Metric 3 — groundedness.
        # ------------------------------------------------------------

        baseline_grounded = groundedness(
            llm,
            baseline_answer,
            baseline_eval_docs,
        )

        advanced_grounded = groundedness(
            llm,
            advanced_answer,
            advanced_eval_docs,
        )

        # ------------------------------------------------------------
        # Accumulate shared evaluation metrics.
        # ------------------------------------------------------------

        baseline_passes += baseline_verdict == "pass"
        advanced_passes += advanced_verdict == "pass"

        baseline_relevance_total += baseline_relevance
        advanced_relevance_total += advanced_relevance

        baseline_grounded_passes += baseline_grounded == "pass"
        advanced_grounded_passes += advanced_grounded == "pass"

        # ------------------------------------------------------------
        # Existing source-diversity metrics.
        # ------------------------------------------------------------

        baseline_sources = _baseline_sources(
            baseline_hits,
            retriever.doc_by_id,
        )

        advanced_sources = _advanced_sources(
            advanced_hits,
            retriever.doc_by_id,
        )

        baseline_source_total += len(baseline_sources)
        advanced_source_total += len(advanced_sources)

        # Advanced-only graph diagnostic.
        graph_context_count = sum(
            1
            for hit in advanced_hits
            if "graph" in hit.get("matched_by", [])
        )

        graph_context_total += graph_context_count

        # Comparison diagnostic.
        overlap_count, overlap_jaccard = _retrieval_overlap(
            baseline_hits,
            advanced_hits,
        )

        # ------------------------------------------------------------
        # Console output.
        # ------------------------------------------------------------

        print("=" * 88)
        print(f"Q{i}: {query}")

        print(f"\nDecomposed into ({len(sub_queries)}):")
        for j, sub_query in enumerate(sub_queries, 1):
            print(f"  {j}. {sub_query}")

        print("\nINTERMEDIATE DECOMPOSITION TOP-5")

        for hit in decomposition_hits:
            doc = retriever.doc_by_id[str(hit["doc_id"])]

            print(
                f"  {hit['score']:.3f} aggregate | "
                f"{doc.metadata['source']} | "
                f"chunk {doc.metadata['chunk']} | "
                f"via {','.join(hit.get('matched_by', []))}"
            )

        # ------------------------------------------------------------
        # Baseline metric block.
        # ------------------------------------------------------------

        print(f"\nBASELINE — {baseline_verdict.upper()}")
        print(f"Answer correctness: {baseline_verdict.upper()}")

        print(
            f"Retrieval relevance@{baseline_retrieved}: "
            f"{baseline_relevant}/{baseline_retrieved} "
            f"({baseline_relevance:.3f})"
        )

        print(f"Groundedness: {baseline_grounded.upper()}")
        print(f"Unique source papers: {len(baseline_sources)}")

        for doc_id, score in baseline_hits:
            doc = retriever.doc_by_id[doc_id]

            print(
                f"  {score:.3f} | "
                f"{doc.metadata['source']} | "
                f"chunk {doc.metadata['chunk']}"
            )

        # ------------------------------------------------------------
        # Advanced metric block.
        # ------------------------------------------------------------

        print(
            f"\nADVANCED (DECOMPOSITION + GRAPH) — "
            f"{advanced_verdict.upper()}"
        )

        print(f"Answer correctness: {advanced_verdict.upper()}")

        print(
            f"Retrieval relevance@{advanced_retrieved}: "
            f"{advanced_relevant}/{advanced_retrieved} "
            f"({advanced_relevance:.3f})"
        )

        print(f"Groundedness: {advanced_grounded.upper()}")
        print(f"Unique source papers: {len(advanced_sources)}")
        print(f"Graph-added context chunks: {graph_context_count}")

        print(
            f"Overlap with baseline: "
            f"{overlap_count}/{TOP_K} chunks; "
            f"Jaccard={overlap_jaccard:.3f}"
        )

        for hit in advanced_hits:
            doc = retriever.doc_by_id[str(hit["doc_id"])]
            matched = ",".join(hit.get("matched_by", []))
            topics = ", ".join(hit.get("shared_topics", []))

            topic_text = (
                f" | topics={topics}"
                if topics
                else ""
            )

            print(
                f"  {hit['score']:.3f} | "
                f"{str(hit.get('role', 'context')).upper():7s} | "
                f"{doc.metadata['source']} | "
                f"chunk {doc.metadata['chunk']} | "
                f"via {matched}{topic_text}"
            )

        print(
            f"\nBASELINE ANSWER:\n"
            f"{baseline_answer}\n"
        )

        print(
            f"ADVANCED ANSWER:\n"
            f"{advanced_answer}\n"
        )

        # ------------------------------------------------------------
        # Log all evidence and metrics for the worksheet/report.
        # ------------------------------------------------------------

        log_text = (
            f"grading_notes:\n{grading_notes}\n\n"

            f"sub_queries:\n"
            f"{json.dumps(sub_queries, indent=2)}\n\n"

            "decomposition_hits:\n"
            + "\n".join(
                f"{hit['score']:.3f} aggregate | "
                f"{retriever.doc_by_id[str(hit['doc_id'])].metadata['source']} | "
                f"chunk "
                f"{retriever.doc_by_id[str(hit['doc_id'])].metadata['chunk']} | "
                f"via {','.join(hit.get('matched_by', []))}"
                for hit in decomposition_hits
            )

            # Baseline metrics.
            + f"\n\nbaseline_verdict={baseline_verdict}\n"
            f"baseline_relevance="
            f"{baseline_relevant}/{baseline_retrieved} "
            f"({baseline_relevance:.3f})\n"
            f"baseline_groundedness={baseline_grounded}\n"
            f"baseline_unique_sources={len(baseline_sources)}\n"

            "baseline_hits:\n"
            + "\n".join(
                f"{score:.3f} | "
                f"{retriever.doc_by_id[doc_id].metadata['source']} | "
                f"chunk {retriever.doc_by_id[doc_id].metadata['chunk']}"
                for doc_id, score in baseline_hits
            )

            + f"\n\nbaseline_answer:\n"
            f"{baseline_answer}\n\n"

            # Advanced metrics.
            f"advanced_verdict={advanced_verdict}\n"
            f"advanced_relevance="
            f"{advanced_relevant}/{advanced_retrieved} "
            f"({advanced_relevance:.3f})\n"
            f"advanced_groundedness={advanced_grounded}\n"
            f"advanced_unique_sources={len(advanced_sources)}\n"
            f"graph_context_chunks={graph_context_count}\n"
            f"baseline_advanced_overlap="
            f"{overlap_count}/{TOP_K}\n"
            f"baseline_advanced_jaccard="
            f"{overlap_jaccard:.3f}\n"

            "advanced_hits:\n"
            + "\n".join(
                f"{hit['score']:.3f} | "
                f"{str(hit.get('role', 'context')).upper()} | "
                f"{retriever.doc_by_id[str(hit['doc_id'])].metadata['source']} | "
                f"chunk "
                f"{retriever.doc_by_id[str(hit['doc_id'])].metadata['chunk']} | "
                f"via {','.join(hit.get('matched_by', []))} | "
                f"topics={','.join(hit.get('shared_topics', []))}"
                for hit in advanced_hits
            )

            + f"\n\nadvanced_answer:\n"
            f"{advanced_answer}"
        )

        log_response(
            f"Q{i}",
            query,
            log_text,
        )

    # ================================================================
    # Overall comparison across all evaluation queries.
    # ================================================================

    n = len(eval_set)

    baseline_avg_sources = (
        baseline_source_total / n
        if n
        else 0.0
    )

    advanced_avg_sources = (
        advanced_source_total / n
        if n
        else 0.0
    )

    graph_avg_context = (
        graph_context_total / n
        if n
        else 0.0
    )

    baseline_avg_relevance = (
        baseline_relevance_total / n
        if n
        else 0.0
    )

    advanced_avg_relevance = (
        advanced_relevance_total / n
        if n
        else 0.0
    )

    print("=" * 88)
    print("COMPARATIVE SUMMARY")

    # Same evaluation metrics for both systems.
    print(
        f"Answer correctness pass rate: "
        f"baseline={baseline_passes}/{n}, "
        f"advanced={advanced_passes}/{n}"
    )

    print(
        f"Average retrieval relevance@{TOP_K}: "
        f"baseline={baseline_avg_relevance:.3f}, "
        f"advanced={advanced_avg_relevance:.3f}"
    )

    print(
        f"Groundedness pass rate: "
        f"baseline={baseline_grounded_passes}/{n}, "
        f"advanced={advanced_grounded_passes}/{n}"
    )

    print(
        "Average unique source papers in top-k: "
        f"baseline={baseline_avg_sources:.2f}, "
        f"advanced={advanced_avg_sources:.2f}"
    )

    # Advanced-only diagnostic.
    print(
        f"Average graph-added context chunks: "
        f"{graph_avg_context:.2f}"
    )

    print(
        "Note: decomposition aggregate scores and graph-context scores are "
        "produced by different ranking stages, so they should not be compared "
        "numerically to the baseline's single-query hybrid score."
    )

    summary = (
        f"baseline_correctness_pass_rate="
        f"{baseline_passes}/{n}\n"

        f"advanced_correctness_pass_rate="
        f"{advanced_passes}/{n}\n"

        f"baseline_avg_retrieval_relevance="
        f"{baseline_avg_relevance:.3f}\n"

        f"advanced_avg_retrieval_relevance="
        f"{advanced_avg_relevance:.3f}\n"

        f"baseline_groundedness_pass_rate="
        f"{baseline_grounded_passes}/{n}\n"

        f"advanced_groundedness_pass_rate="
        f"{advanced_grounded_passes}/{n}\n"

        f"baseline_avg_unique_sources="
        f"{baseline_avg_sources:.2f}\n"

        f"advanced_avg_unique_sources="
        f"{advanced_avg_sources:.2f}\n"

        f"average_graph_context_chunks="
        f"{graph_avg_context:.2f}"
    )

    log_response(
        "COMPARATIVE SUMMARY",
        "all evaluation queries",
        summary,
    )

# real-system entry point replacing the starter demo.
def main() -> None:
    llm = make_llm()

    print("Advanced-retrieval plan:")
    print(json.dumps(my_advanced_plan(), indent=2))
    print()

    docs, db = load_corpus_and_db()

    retriever = HybridRetriever(
        docs=docs,
        db=db,
    )

    graph = build_graph(docs)
    print(
        f"Graph ready: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges."
    )

    run_comparison(llm, graph, retriever)

    print(f"\nResults saved to {LOG_PATH}")
    print(f"Decomposition cache: {DECOMPOSITION_CACHE_PATH}")
    print(f"Graph-topic cache: {TOPIC_CACHE_PATH}")


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
