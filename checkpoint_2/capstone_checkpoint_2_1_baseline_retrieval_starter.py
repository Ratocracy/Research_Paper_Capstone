r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TEMPERATURE = 0.2
TOP_K = 5
CANDIDATE_POOL = 15

# Equal weighting for baseline hybrid retrieval
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

# Simple character-based chunking
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500

CHROMA_DIR = Path(__file__).resolve().parent / "checkpoint_2_1_chroma"
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
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


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# There are six short "documents" on distinct topics so a baseline retriever has something to
# discriminate between. Your real corpus is the PDFs/articles in `Labs/CapstoneDatasets/`,
# which came with the course in Module 1. Point your code at your local copy of that
# folder, and update the path if your checkout puts it elsewhere.


# %%
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long paper text into overlapping chunks."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try not to cut directly through a word.
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
    """Read PDFs and convert them into chunk-level LangChain Documents."""

    docs = []

    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in: {corpus_dir}"
        )

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))

        paper_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

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

    print(
        f"Loaded {len(pdf_files)} papers "
        f"into {len(docs)} chunks."
    )

    return docs


def build_or_load_db(docs: list[Document]) -> Chroma:
    """Build the vector index once, then reuse it on future runs."""

    if CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir()):
        print(f"Loading vector DB from {CHROMA_DIR}")

        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
        )

    print("Building vector DB — embedding research paper chunks...")

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        ids=[d.metadata["id"] for d in docs],
    )


# %% [markdown]
# ## Step 2 — Hybrid baseline retriever
#
# The baseline combines BM25 keyword retrieval with Chroma vector retrieval.
# Scores from each retriever are normalized and combined using equal weights.
# This preserves exact-term matching while also supporting semantic similarity.

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

def _normalize(
    scores: list[float],
    invert: bool = False,
) -> list[float]:

    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if high == low:
        return [0.5] * len(scores)

    normalized = [
        (score - low) / (high - low)
        for score in scores
    ]

    if invert:
        normalized = [1.0 - score for score in normalized]

    return normalized

class HybridRetriever:
    def __init__(self, docs: list[Document], db: Chroma):
        self.docs = docs
        self.db = db

        self.doc_by_id = {
            doc.metadata["id"]: doc
            for doc in docs
        }

        self.ids = [
            doc.metadata["id"]
            for doc in docs
        ]

        self.bm25 = BM25Okapi(
            [_tokens(doc.page_content) for doc in docs]
        )

    def _bm25_topk(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, float]]:

        scores = self.bm25.get_scores(_tokens(query))

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self.ids[i], float(scores[i]))
            for i in indices
        ]

    def _vector_topk(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, float]]:

        results = self.db.similarity_search_with_score(
            query,
            k=k,
        )

        return [
            (
                doc.metadata["id"],
                float(distance),
            )
            for doc, distance in results
        ]

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
    ) -> list[tuple[str, float]]:

        bm25_results = self._bm25_topk(
            query,
            CANDIDATE_POOL,
        )

        vector_results = self._vector_topk(
            query,
            CANDIDATE_POOL,
        )

        bm25_scores = _normalize(
            [score for _, score in bm25_results]
        )

        # Chroma gives distance:
        # LOW distance = GOOD result.
        vector_scores = _normalize(
            [score for _, score in vector_results],
            invert=True,
        )

        candidates = {}

        for (doc_id, _), score in zip(
            bm25_results,
            bm25_scores,
        ):
            candidates.setdefault(
                doc_id,
                {"bm25": 0.0, "vector": 0.0},
            )

            candidates[doc_id]["bm25"] = score

        for (doc_id, _), score in zip(
            vector_results,
            vector_scores,
        ):
            candidates.setdefault(
                doc_id,
                {"bm25": 0.0, "vector": 0.0},
            )

            candidates[doc_id]["vector"] = score

        fused = []

        for doc_id, scores in candidates.items():
            fused_score = (
                WEIGHT_BM25 * scores["bm25"]
                + WEIGHT_VECTOR * scores["vector"]
            )

            fused.append(
                (doc_id, fused_score)
            )

        fused.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return fused[:k]

def answer(
    llm: ChatOpenAI,
    query: str,
    doc_ids: list[str],
    doc_by_id: dict[str, Document],
) -> str:

    context_parts = []

    for doc_id in doc_ids:
        doc = doc_by_id[doc_id]

        source = doc.metadata["source"]
        chunk = doc.metadata["chunk"]

        context_parts.append(
            f"[Source: {source} | Chunk: {chunk}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"Documents:\n{context}\n\n"
                f"Question: {query}"
            )
        ),
    ]

    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Return 3-5 representative queries for YOUR chosen scenario.

    TODO — your turn. See the guidance above. Each item is a query string. Pick
    queries that a real user of your system would ask and that require different
    retrieval behaviors (single-doc, multi-doc, paraphrased).

    Delete the raise NotImplementedError line once your code works.
    """
    return [
        "What is program synthesis and what problem is it intended to solve?",

        "How does the sketching approach to program synthesis work?",

        "How do program synthesis systems use specifications or examples to generate programs?",

        "Compare the major approaches to program synthesis discussed across the papers.",

        "What limitations or open problems do the papers identify for program synthesis?",
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%

def debug_retrieval(retriever: HybridRetriever, query: str, k: int = 5) -> None:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")

    print("\nBM25 ONLY")
    for doc_id, score in retriever._bm25_topk(query, k):
        doc = retriever.doc_by_id[doc_id]
        print(
            f"{score:.3f} | "
            f"{doc.metadata['source']} | "
            f"chunk {doc.metadata['chunk']}"
        )

    print("\nVECTOR ONLY")
    for doc_id, distance in retriever._vector_topk(query, k):
        doc = retriever.doc_by_id[doc_id]
        print(
            f"{distance:.3f} distance | "
            f"{doc.metadata['source']} | "
            f"chunk {doc.metadata['chunk']}"
        )

    print("\nHYBRID")
    for doc_id, score in retriever.retrieve(query, k):
        doc = retriever.doc_by_id[doc_id]
        print(
            f"{score:.3f} | "
            f"{doc.metadata['source']} | "
            f"chunk {doc.metadata['chunk']}"
        )


def run() -> None:
    llm = make_llm()

    if len(sys.argv) > 1:
        corpus_dir = Path(sys.argv[1])
    else:
        corpus_dir = (
            Path(__file__).resolve().parent.parent
            / "ResearchPapers"
        )

    if not corpus_dir.exists():
        raise RuntimeError(
            f"Research paper directory not found: {corpus_dir}\n"
            "Pass its path as the first command-line argument."
        )

    docs = load_research_papers(corpus_dir)

    db = build_or_load_db(docs)

    retriever = HybridRetriever(
        docs=docs,
        db=db,
    )

    # Temporary diagnostic tests
    debug_retrieval(
        retriever,
        "How does the sketching approach to program synthesis work?"
    )

    debug_retrieval(
        retriever,
        "How can a partially written program have its missing implementation details automatically completed?"
    )

    queries = my_representative_queries()

    print(
        f"\nCheckpoint 2.1 — hybrid retrieval"
        f" | scenario: {SCENARIO}\n"
    )

    for i, query in enumerate(queries, 1):
        hits = retriever.retrieve(query, TOP_K)

        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print("\nRetrieved:")

        evidence = []

        for doc_id, score in hits:
            doc = retriever.doc_by_id[doc_id]

            source = doc.metadata["source"]
            chunk = doc.metadata["chunk"]

            line = (
                f"  {score:.3f} | "
                f"{source} | chunk {chunk}"
            )

            print(line)
            evidence.append(line)

        if not hits:
            print("No relevant documents retrieved.")
            continue

        response = answer(
            llm,
            query,
            [doc_id for doc_id, _ in hits],
            retriever.doc_by_id,
        )

        print(f"\nANSWER:\n{response}\n")

        log(
            f"QUERY {i}: {query}",
            (
                "retrieved:\n"
                + "\n".join(evidence)
                + f"\n\nanswer:\n{response}"
            ),
        )

    print("=" * 72)
    print(
        "Done. Results saved to "
        f"{LOG_PATH}"
    )


run()
# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
