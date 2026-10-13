r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ADDED FOR CHECKPOINT 5.1
# Reuse the completed Checkpoint 4 retrieval system.
SCRIPT_DIR = Path(__file__).resolve().parent
COURSE_ROOT = SCRIPT_DIR.parent
CHECKPOINT_4_DIR = COURSE_ROOT / "checkpoint_4"

if str(CHECKPOINT_4_DIR) not in sys.path:
    sys.path.insert(0, str(CHECKPOINT_4_DIR))

import capstone_checkpoint_4_1_advanced_retrieval_starter as cp4

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 4
AGENT_CONTEXT_K = 12
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are the controller for a research-paper RAG agent. Given the user's question, "
    "the retrieval actions already taken, and the evidence collected so far, choose the "
    "next action. Available actions are: hybrid_search, decomposed_search, graph_expand, "
    "or answer. "

    "Use hybrid_search for a focused follow-up query. "
    "Use decomposed_search when the question has multiple parts that are not all covered. "
    "Use graph_expand when the current evidence is relevant but related papers may add "
    "useful context. Choose answer once the evidence directly supports every major part "
    "of the question. Do not repeat the same action/query pair. "

    "Before choosing answer, identify every distinct part of the user's question. "
    "Only answer when the collected evidence directly supports each part. "
    "For requests asking for limitations, comparisons, causes, or multiple examples, "
    "do not stop after finding only one supported item unless the user asked for one. "

    "Return ONLY JSON in this form: "
    '{"action":"hybrid_search|decomposed_search|graph_expand|answer", '
    '"query":"...", "reasoning":"..."}. '
    "For answer, query may be an empty string."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
#
# Instead of the starter's tiny demonstration corpus, this section reuses the full
# Checkpoint 4 Research Paper Navigator. The agent can choose among the existing
# hybrid, decomposed, and graph-expanded retrieval capabilities before answering.

# %%
def _merge_hits(
    collected: dict[str, dict[str, Any]],
    new_hits: list[dict[str, Any]],
) -> None:
    """Add newly retrieved chunks without duplicating existing evidence."""

    for hit in new_hits:
        doc_id = str(hit["doc_id"])

        if doc_id in collected:
            existing = collected[doc_id]

            existing["matched_by"] = sorted(
                set(existing.get("matched_by", []))
                | set(hit.get("matched_by", []))
            )

            existing["shared_topics"] = sorted(
                set(existing.get("shared_topics", []))
                | set(hit.get("shared_topics", []))
            )
            continue

        if len(collected) >= AGENT_CONTEXT_K:
            continue

        collected[doc_id] = {
            **hit,
            "doc_id": doc_id,
            "shared_topics": list(hit.get("shared_topics", [])),
        }


def _hybrid_hits(
    retriever: cp4.HybridRetriever,
    query: str,
) -> list[dict[str, Any]]:
    """Wrap Checkpoint 2 hybrid results in the Checkpoint 4 hit format."""

    return [
        {
            "doc_id": doc_id,
            "score": score,
            "role": "primary",
            "matched_by": ["hybrid"],
            "shared_topics": [],
        }
        for doc_id, score in retriever.retrieve(query, cp4.TOP_K)
    ]


def _run_tool(
    llm: ChatOpenAI,
    action: str,
    query: str,
    graph,
    retriever: cp4.HybridRetriever,
    collected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Execute one retrieval tool and return hits plus extra LLM-call count."""

    if action == "hybrid_search":
        return _hybrid_hits(retriever, query), 0

    if action == "decomposed_search":
        hits, _sub_queries = cp4.multistep_retrieve(
            llm,
            retriever,
            query,
            cp4.TOP_K,
        )

        for hit in hits:
            hit.setdefault("shared_topics", [])

        # decompose_query() makes one LLM call.
        return hits, 1

    if action == "graph_expand":
        # Use the most recently collected evidence as graph seeds.
        seeds = list(collected.values())[-cp4.GRAPH_SEED_K:]

        hits = cp4.graph_retrieve(
            graph,
            query,
            seeds,
            retriever,
            context_k=cp4.GRAPH_CONTEXT_SLOTS,
        )
        return hits, 0

    return [], 0


def decide(
    llm: ChatOpenAI,
    question: str,
    collected: dict[str, dict[str, Any]],
    executed: list[dict[str, str]],
    retriever: cp4.HybridRetriever,
) -> dict[str, str]:
    """Let the agent decide which retrieval action to take next."""

    evidence_parts = []

    for i, (doc_id, _hit) in enumerate(collected.items(), 1):
        doc = retriever.doc_by_id[doc_id]
        evidence_parts.append(
            f"[{i}] {doc.metadata['source']} chunk {doc.metadata['chunk']}\n"
            f"{doc.page_content[:1200]}"
        )

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(none yet)"

    user = (
        f"Question:\n{question}\n\n"
        f"Actions already taken:\n{json.dumps(executed, indent=2)}\n\n"
        f"Evidence so far:\n{evidence}"
    )

    raw = llm.invoke(
        [
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=user),
        ]
    ).content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

    try:
        decision = json.loads(raw)
        action = str(decision.get("action", "answer"))
        query = str(decision.get("query", "")).strip()
        reasoning = str(decision.get("reasoning", ""))

        allowed_actions = {
            "hybrid_search",
            "decomposed_search",
            "graph_expand",
            "answer",
        }

        if action not in allowed_actions:
            action = "answer"

        return {
            "action": action,
            "query": query,
            "reasoning": reasoning,
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "action": "answer",
            "query": "",
            "reasoning": "parse-fail -> answer",
        }


def agentic_answer(
    llm: ChatOpenAI,
    question: str,
    graph,
    retriever: cp4.HybridRetriever,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], int]:
    """Agentic loop: retrieve -> inspect -> choose next tool -> repeat -> answer."""

    collected: dict[str, dict[str, Any]] = {}
    executed: list[dict[str, str]] = []
    trace: list[dict[str, Any]] = []
    llm_calls = 0

    # Start with the same hybrid retrieval used by the baseline.
    action = "hybrid_search"
    query = question

    for step in range(1, MAX_STEPS + 1):
        new_hits, extra_calls = _run_tool(
            llm,
            action,
            query,
            graph,
            retriever,
            collected,
        )
        llm_calls += extra_calls

        _merge_hits(collected, new_hits)

        executed.append(
            {
                "action": action,
                "query": query,
            }
        )

        # After seeing the new evidence, the agent decides what comes next.
        decision = decide(
            llm,
            question,
            collected,
            executed,
            retriever,
        )
        llm_calls += 1

        # Hard safety against repeating an identical action/query pair.
        proposed_query = decision["query"] or question
        if decision["action"] != "answer" and any(
            x["action"] == decision["action"] and x["query"] == proposed_query
            for x in executed
        ):
            decision = {
                "action": "answer",
                "query": "",
                "reasoning": "Stopped to avoid repeating an identical retrieval action/query.",
            }

        trace.append(
            {
                "step": step,
                "action": action,
                "query": query,
                "new_hits": len(new_hits),
                "total_docs": len(collected),
                "decision": decision,
            }
        )

        print(
            f"           next={decision['action']} | "
            f"{decision['reasoning'][:100]}"
        )

        # Normal stop: the agent believes it has enough evidence.
        if decision["action"] == "answer":
            break

        # Budget stop: the agent wants another retrieval step,
        # but the maximum number of steps has been reached.
        if step == MAX_STEPS:
            print(
                "           MAX_STEPS reached — "
                "generating best-effort answer from collected evidence."
            )

            trace[-1]["budget_exhausted"] = True
            break

        action = decision["action"]

        query = (
            decision["query"]
            or question
        )

    final_hits = list(collected.values())

    answer = cp4.answer_from_docs(
        llm,
        question,
        final_hits,
        retriever.doc_by_id,
    )
    llm_calls += 1

    return answer, trace, final_hits, llm_calls


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "tools": [
            "hybrid_search",
            "decomposed_search",
            "graph_expand",
            "answer",
        ],
        "stop_condition": (
            "Answer when the retrieved evidence directly supports every major "
            "part of the question; otherwise choose another targeted retrieval action."
        ),
        "system_prompt_idea": (
            "Start with hybrid retrieval, then inspect the evidence. Use decomposition "
            "for uncovered sub-parts, graph expansion for related cross-paper context, "
            "and answer only when the evidence is sufficient."
        ),
        "test_tasks": [
            "Compare the major approaches to program synthesis discussed across the papers.",
            "What limitations or open problems do the papers identify for program synthesis?",
            "How does the sketching approach work, and what limitations remain for program synthesis?",
        ],
    }




# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def my_agent_eval_set() -> list[dict[str, str]]:
    """Representative multistep tasks for baseline vs. agent comparison."""

    return [
        {
            "question": (
                "Compare the major approaches to program synthesis "
                "discussed across the papers."
            ),
            "grading_notes": (
                "Correctly distinguishes at least three supported synthesis approaches, "
                "such as syntax-guided synthesis, SMT or refutation-based synthesis, "
                "adaptive concretization, or programmable synthesis."
            ),
        },
        {
            "question": (
                "What limitations or open problems do the papers "
                "identify for program synthesis?"
            ),
            "grading_notes": (
                "Identifies at least two supported limitations, such as computational "
                "difficulty or large search spaces and usability or tight coupling "
                "between specifications and synthesis solvers."
            ),
        },
        {
            "question": (
                "How does the sketching approach work, and what "
                "limitations remain for program synthesis?"
            ),
            "grading_notes": (
                "Explains that sketching uses a partial program whose missing details "
                "are filled by synthesis subject to a specification, and identifies at "
                "least two supported limitations or open problems for program synthesis."
            ),
        },
    ]


def run_comparison(
    llm: ChatOpenAI,
    graph,
    retriever: cp4.HybridRetriever,
) -> None:
    """Compare the fixed hybrid baseline with the agentic workflow."""

    baseline_correct = 0
    agent_correct = 0
    baseline_relevance_sum = 0.0
    agent_relevance_sum = 0.0
    baseline_grounded_count = 0
    agent_grounded_count = 0
    baseline_runtime_sum = 0.0
    agent_runtime_sum = 0.0
    agent_workflow_calls_sum = 0

    eval_set = my_agent_eval_set()

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        grading_notes = item["grading_notes"]

        print("\n" + "=" * 88)
        print(f"Q{i}: {question}\n")

        # BASELINE — fixed single-pass hybrid retrieval.
        baseline_start = time.perf_counter()
        baseline_hits = retriever.retrieve(question, cp4.TOP_K)
        baseline_answer = cp4.answer_from_baseline(
            llm,
            question,
            baseline_hits,
            retriever.doc_by_id,
        )
        baseline_seconds = time.perf_counter() - baseline_start

        # AGENT — retrieval strategy selected dynamically.
        agent_start = time.perf_counter()
        agent_answer, trace, agent_hits, agent_calls = agentic_answer(
            llm,
            question,
            graph,
            retriever,
        )
        agent_seconds = time.perf_counter() - agent_start

        # Convert evidence to Documents for identical evaluation metrics.
        baseline_docs = [
            retriever.doc_by_id[doc_id]
            for doc_id, _ in baseline_hits
        ]
        agent_docs = [
            retriever.doc_by_id[str(hit["doc_id"])]
            for hit in agent_hits
        ]

        # Same Checkpoint 4 answer-correctness judge.
        baseline_verdict = cp4.judge(llm, baseline_answer, grading_notes)
        agent_verdict = cp4.judge(llm, agent_answer, grading_notes)

        # Same Checkpoint 4 retrieval relevance metric.
        (
            baseline_relevant,
            baseline_total,
            baseline_relevance,
        ) = cp4.retrieval_relevance(llm, question, baseline_docs)

        (
            agent_relevant,
            agent_total,
            agent_relevance,
        ) = cp4.retrieval_relevance(llm, question, agent_docs)

        # Same Checkpoint 4 groundedness metric.
        baseline_grounded = cp4.groundedness(llm, baseline_answer, baseline_docs)
        agent_grounded = cp4.groundedness(llm, agent_answer, agent_docs)

        # Accumulate summary metrics.
        baseline_correct += baseline_verdict == "pass"
        agent_correct += agent_verdict == "pass"
        baseline_relevance_sum += baseline_relevance
        agent_relevance_sum += agent_relevance
        baseline_grounded_count += baseline_grounded == "pass"
        agent_grounded_count += agent_grounded == "pass"
        baseline_runtime_sum += baseline_seconds
        agent_runtime_sum += agent_seconds
        agent_workflow_calls_sum += agent_calls

        print("\nBASELINE")
        print(f"Answer correctness: {baseline_verdict.upper()}")
        print(
            f"Retrieval relevance: {baseline_relevant}/{baseline_total} "
            f"({baseline_relevance:.3f})"
        )
        print(f"Groundedness: {baseline_grounded.upper()}")
        print(f"Runtime: {baseline_seconds:.2f} seconds")
        print("Answer-generation LLM calls: 1")

        print("\nAGENT")
        print(f"Answer correctness: {agent_verdict.upper()}")
        print(
            f"Retrieval relevance: {agent_relevant}/{agent_total} "
            f"({agent_relevance:.3f})"
        )
        print(f"Groundedness: {agent_grounded.upper()}")
        print(f"Runtime: {agent_seconds:.2f} seconds")
        print(f"Workflow LLM calls: {agent_calls}")

        print("\nAGENT TRACE")
        for step in trace:
            print(
                f"  Step {step['step']}: {step['action']} | "
                f"query={step['query']}"
            )
            print(
                f"    Added {step['new_hits']} chunks; "
                f"total evidence={step['total_docs']}"
            )
            decision = step["decision"]
            print(
                f"    Decision: {decision['action']} | "
                f"{decision['reasoning']}"
            )

        print(f"\nBASELINE ANSWER:\n{baseline_answer}\n")
        print(f"AGENT ANSWER:\n{agent_answer}\n")

        log(
            f"Q{i}",
            (
                f"QUESTION:\n{question}\n\n"
                f"BASELINE correctness={baseline_verdict}\n"
                f"BASELINE relevance={baseline_relevance:.3f}\n"
                f"BASELINE groundedness={baseline_grounded}\n"
                f"BASELINE runtime={baseline_seconds:.2f}\n\n"
                f"AGENT correctness={agent_verdict}\n"
                f"AGENT relevance={agent_relevance:.3f}\n"
                f"AGENT groundedness={agent_grounded}\n"
                f"AGENT runtime={agent_seconds:.2f}\n"
                f"AGENT workflow_llm_calls={agent_calls}\n\n"
                f"AGENT TRACE:\n{json.dumps(trace, indent=2)}\n\n"
                f"BASELINE ANSWER:\n{baseline_answer}\n\n"
                f"AGENT ANSWER:\n{agent_answer}"
            ),
        )

    n = len(eval_set)
    print("\n" + "=" * 88)
    print("COMPARATIVE SUMMARY")
    print(
        f"Answer correctness pass rate: baseline={baseline_correct}/{n}, "
        f"agent={agent_correct}/{n}"
    )
    print(
        f"Average retrieval relevance: baseline={baseline_relevance_sum / n:.3f}, "
        f"agent={agent_relevance_sum / n:.3f}"
    )
    print(
        f"Groundedness pass rate: baseline={baseline_grounded_count}/{n}, "
        f"agent={agent_grounded_count}/{n}"
    )
    print(
        f"Average runtime: baseline={baseline_runtime_sum / n:.2f}s, "
        f"agent={agent_runtime_sum / n:.2f}s"
    )
    print(
        f"Average agent workflow LLM calls: "
        f"{agent_workflow_calls_sum / n:.2f}"
    )


def run() -> None:
    llm = make_llm()

    print("Checkpoint 5.1 — Agentic RAG")
    print(f"Scenario: {SCENARIO}\n")

    print("Agent plan:")
    print(json.dumps(my_agent_plan(), indent=2))
    print()

    # Reuse the full Checkpoint 4 corpus/vector database.
    docs, db = cp4.load_corpus_and_db()

    retriever = cp4.HybridRetriever(
        docs=docs,
        db=db,
    )

    # Reuse the Checkpoint 4 paper/chunk/topic graph.
    graph = cp4.build_graph(docs)

    print(
        f"Graph ready: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges.\n"
    )

    run_comparison(llm, graph, retriever)

    print(f"\nResults saved to {LOG_PATH}")


if __name__ == "__main__":
    run()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
