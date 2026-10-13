"""Interactive, independent questions over the Research Paper Navigator corpus.

From the repository root, run:
    python checkpoint_8/interactive_qa.py

Uses the existing OPENROUTER_API_KEY configuration and corpus/index discovery.
An optional first argument supplies the ResearchPapers directory when building
an index. Initial indexing and uncached questions make paid API calls.
"""

import sys
import time
from pathlib import Path

# import checkpoint
CHECKPOINT_7_DIR = Path(__file__).resolve().parent.parent / "checkpoint_7"
if str(CHECKPOINT_7_DIR) not in sys.path:
    sys.path.insert(0, str(CHECKPOINT_7_DIR))

import capstone_checkpoint_7_1_production_ready_starter as navigator


def question_loop(graph, retriever) -> None:
    """Reuse the loaded corpus and session cache across independent questions."""
    cache = navigator.SimpleCache()
    print("\nResearch Paper Navigator")
    print("Ask a question about the papers. Each question is independent;")
    print("previous questions and answers are not sent as conversation history.")
    print("Commands: /quit to exit, /clear to clear cached answers.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in {"/quit", "/exit"}:
            print("Goodbye.")
            return
        if question.lower() == "/clear":
            cache = navigator.SimpleCache()
            print("Answer cache cleared.")
            continue

        started = time.perf_counter()
        try:
            answer, usage, route, _ = navigator.real_answer_routed(
                question, cache, graph, retriever
            )
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return
        except Exception as exc:
            print(f"Could not answer ({type(exc).__name__}). Check your API access "
                  "and network connection, then try again.", file=sys.stderr)
            continue

        print(f"\nNavigator: {answer}")
        print(
            f"\n[Path: {route} | Model calls: {usage['total_calls']} | "
            f"Model tokens: {usage['total_tokens']} | "
            f"Time: {time.perf_counter() - started:.2f}s]"
        )


def main() -> int:
    try:
        navigator.check_api_key()
        print("Loading the research papers and retrieval index...")
        print("If no reusable index exists, initial indexing makes embedding API calls.")
        retrieval = navigator.cp6.cp5.cp4
        docs, db = retrieval.load_corpus_and_db()
        if not docs:
            print("No paper chunks were loaded. Check your corpus.", file=sys.stderr)
            return 1
        retriever = retrieval.HybridRetriever(docs=docs, db=db)
        graph = retrieval.build_graph(docs)
        print(f"Ready: {len(docs)} paper chunks.")
        question_loop(graph, retriever)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0
    except Exception as exc:
        print(
            f"Setup failed ({type(exc).__name__}). Check dependencies, "
            "OPENROUTER_API_KEY, network access, and the ResearchPapers directory.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
