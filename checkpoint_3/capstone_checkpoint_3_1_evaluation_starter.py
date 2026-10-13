r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")

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

LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# Reuse the Checkpoint 2.1 Chroma DB when the expected sibling folder exists.
# Otherwise create a local Checkpoint 3.1 copy using the same embeddings/corpus.
SCRIPT_DIR = Path(__file__).resolve().parent
COURSE_ROOT = SCRIPT_DIR.parent
CHECKPOINT_2_CHROMA = COURSE_ROOT / "checkpoint_2" / "checkpoint_2_1_chroma"
LOCAL_CHROMA = SCRIPT_DIR / "checkpoint_3_1_chroma"

CHROMA_DIR = (
    CHECKPOINT_2_CHROMA
    if CHECKPOINT_2_CHROMA.is_dir() and any(CHECKPOINT_2_CHROMA.iterdir())
    else LOCAL_CHROMA
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
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

def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Demonstration corpus + retriever (provided to illustrate the evaluation workflow)

# %%
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long paper text into overlapping character-based chunks."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Avoid cutting directly through a word when possible.
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
    """Read research-paper PDFs and convert them to chunk-level Documents."""

    docs: list[Document] = []
    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {corpus_dir}")

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
    """Load the persisted vector index when available, otherwise build it."""

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
# ## Checkpoint 2.1 hybrid baseline retriever

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
    """Equal-weight BM25 + vector-search hybrid retriever from Checkpoint 2.1."""

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

        # Chroma returns distance: lower distance = better match.
        vector_scores = _normalize(
            [score for _, score in vector_results],
            invert=True,
        )

        candidates: dict[str, dict[str, float]] = {}

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

        fused: list[tuple[str, float]] = []

        for doc_id, scores in candidates.items():
            fused_score = (
                WEIGHT_BM25 * scores["bm25"]
                + WEIGHT_VECTOR * scores["vector"]
            )

            fused.append((doc_id, fused_score))

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
    context_parts: list[str] = []

    for doc_id in doc_ids:
        if doc_id not in doc_by_id:
            continue

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
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, 
          answer_text: str, 
          grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """Return 3-5 evaluation items for YOUR scenario.

    TODO — your turn. Each item is {"question": "...", "grading_notes": "..."}.
    The grading_notes describe what a correct answer MUST contain, in one sentence.
    Include at least one question you expect your baseline to get WRONG, so your
    diagnosis has something to find.

    Delete the raise NotImplementedError line once your code works.
    """
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
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation(
    llm: ChatOpenAI,
    retriever: HybridRetriever,
) -> None:
    eval_set = my_eval_set()
    passes = 0

    print(
        f"\nCheckpoint 3.1 — baseline evaluation"
        f" | scenario: {SCENARIO}\n"
    )

    for i, item in enumerate(eval_set, 1):
        query = item["question"]
        hits = retriever.retrieve(query, TOP_K)

        if hits:
            ans = answer(
                llm,
                query,
                [doc_id for doc_id, _ in hits],
                retriever.doc_by_id,
            )
        else:
            ans = "(no documents retrieved)"

        verdict = judge(
            llm,
            ans,
            item["grading_notes"],
        )

        passes += verdict == "pass"

        print("=" * 72)
        print(f"Q{i}: {query}")
        print(f"Verdict: {verdict.upper()}")
        print("\nRetrieved:")

        evidence: list[str] = []

        if not hits:
            print("  No relevant documents retrieved.")

        for doc_id, score in hits:
            doc = retriever.doc_by_id[doc_id]

            line = (
                f"  {score:.3f} | "
                f"{doc.metadata['source']} | "
                f"chunk {doc.metadata['chunk']}"
            )

            print(line)
            evidence.append(line)

        print(f"\nAnswer:\n{ans}\n")

        log(
            f"Q{i}: {query}",
            (
                "retrieved:\n"
                + ("\n".join(evidence) if evidence else "(none)")
                + f"\n\nverdict={verdict}"
                + f"\n\ngrading_notes:\n{item['grading_notes']}"
                + f"\n\nanswer:\n{ans}"
            ),
        )

    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")

    log(
        "BASELINE SUMMARY",
        f"pass_rate={passes}/{len(eval_set)}",
    )


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework(
    llm: ChatOpenAI,
    retriever: HybridRetriever,
) -> None:
    q = "How does the sketching approach to program synthesis work?"

    notes = (
        "Explains that a programmer provides a partial program or sketch and that "
        "an automated synthesis process fills in missing implementation details "
        "while satisfying a specification or correctness criteria."
    )

    hits = retriever.retrieve(q, TOP_K)

    if not hits:
        good = "(no documents retrieved)"
    else:
        good = answer(
            llm,
            q,
            [doc_id for doc_id, _ in hits],
            retriever.doc_by_id,
        )

    manipulated = (
        "Program sketching is a computer graphics technique where an artist draws "
        "an approximate picture before a rendering system generates the final image."
    )

    good_verdict = judge(llm, good, notes)
    manipulated_verdict = judge(llm, manipulated, notes)

    print("\n--- Framework validation ---")
    print(
        f"  correct answer     -> {good_verdict.upper()} "
        "(expected PASS)"
    )
    print(
        f"  manipulated answer -> {manipulated_verdict.upper()} "
        "(expected FAIL)"
    )
    print(
        "  The framework works if it passes the grounded answer and fails "
        "the deliberately incorrect answer."
    )

    log(
        "FRAMEWORK VALIDATION",
        (
            f"good={good_verdict}\n"
            f"manipulated={manipulated_verdict}\n\n"
            f"manipulated_answer:\n{manipulated}"
        ),
    )

# %%
# %%
def main() -> None:
    llm = make_llm()

    # Same path behavior used in Checkpoint 2.1:
    #   python script.py path/to/ResearchPapers
    # or, by default, ResearchPapers lives in the course root.
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

    run_evaluation(llm, retriever)
    validate_framework(llm, retriever)

    print(f"\nResults saved to {LOG_PATH}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
