# Research Paper Navigator

Research Paper Navigator is a retrieval-augmented LLM system for exploring a local collection of research papers about program synthesis. It helps readers explain concepts, compare approaches, and investigate limitations using evidence retrieved from PDF documents.

The final interface is an interactive terminal application. Questions use live model calls; answers are not prerecorded. Each question is independent, with no conversational history. Earlier checkpoints preserve the system's development and evaluation workflows.

## Install

Download or clone this repository and open a terminal in its root directory. Python 3.11 is the version used by the demonstrated local environment. Internet access and an OpenRouter API key with access to the configured chat and embedding models are required for live use.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux equivalent:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Copy the example configuration only when creating a new `.env`; preserve an existing configured file. Edit `.env` and replace the placeholder with your own `OPENROUTER_API_KEY`. Do not commit `.env`. The root requirements cover the current application and its imported checkpoint modules; the older checkpoint requirements are historical course environment lists.

## Corpus and configuration

Place the research-paper PDFs directly in `ResearchPapers/` at the repository root. The demonstrated collection contained 153 PDFs, producing 5,323 chunks. Different PDFs or extraction versions can produce different counts and answers. The `Wikipedia/` dataset is not used by the final research-paper workflow.

If the PDFs are absent from your checkout, obtain the project's research-paper dataset from the submitter/course materials before running. Source PDFs retain their original rights; this project does not grant permission to redistribute them. A smaller collection can demonstrate Q&A, but will not reproduce the full collection's results.

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Required credential for chat and embedding requests. |
| `CAPSTONE_RESEARCH_PAPERS` | Optional PDF folder used when no reusable data is available; defaults to repository `ResearchPapers/`. |
| `CAPSTONE_CHROMA_DIR` | Optional existing compatible Chroma index to reuse. This does not select the output location for a new index. |

Run from the repository root. Configuration is loaded from `.env`; existing process environment variables take precedence. The code currently configures `openai/gpt-5.4-mini` for planning/answering and `openai/text-embedding-3-small` for embeddings through OpenRouter. These are code constants, not `.env` settings. Provider availability and account access must support these model IDs.

First startup extracts PDFs and builds an embedding index if none is available. This can take time and incurs embedding API usage. Later launches reuse a compatible index, searching checkpoints 2 and 3 before checkpoint 4, unless an existing index override is supplied. The fallback index is written to `checkpoint_4/checkpoint_4_1_chroma/`; extracted chunks are cached in `checkpoint_4/checkpoint_4_1_chunks.jsonl`.

Changing the PDF folder does not override an already discovered index or chunk cache. To rebuild for a changed corpus, move the old generated indexes and chunk cache outside the discovery locations first, or use a separate fresh checkout. Keep the embedding model consistent with the model used to build an index.

## Run interactive Q&A

Windows, without needing virtual-environment activation:

```powershell
.\.venv\Scripts\python.exe checkpoint_8/interactive_qa.py
```

macOS/Linux:

```bash
.venv/bin/python checkpoint_8/interactive_qa.py
```

With your virtual environment activated, the command is simply `python checkpoint_8/interactive_qa.py`. An optional first positional argument supplies a PDF directory when no reusable data is available.

Wait for `Ready: ... paper chunks.`, then enter a question at `You:`. Example questions:

- How does sketching work in program synthesis?
- What is syntax-guided synthesis?
- Compare the major approaches to program synthesis discussed across the papers.
- What limitations or open problems do the papers identify for program synthesis?

`/quit` or `/exit` ends the session; `/clear` clears the session's answer cache. Blank input is ignored. Repeating a normalized question reuses its cached answer. The cache is in memory and disappears on exit. Terminal output shows the route, model calls, model tokens, and elapsed time. These token/call counts cover tracked chat models, not embedding requests or total billing.

## Architecture and workflow

The application imports checkpoint 7, which reuses checkpoints 6, 5, and 4. Keep these directories together.

1. Extract PDF text into overlapping chunks (4,000 characters, 500 overlap), retaining source and chunk metadata.
2. Retrieve candidates using BM25 keyword scoring and Chroma vector similarity, combining normalized scores with equal weights.
3. Sanitize the question before retrieval and check the normalized exact-match answer cache.
4. Route focused questions to one hybrid retrieval and one answer call. Route heuristic matches for complex questions to the agent.
5. The agent can use hybrid search, decomposed queries, and topic-graph expansion. Its loop is capped at four steps and prevents repeated identical action/query pairs.
6. For the complex path, rerank accumulated evidence and limit the final context to six chunks. Generate the answer with escaped document/question XML sections and instructions to use retrieved evidence.

NetworkX supplies the topic graph. LangChain integrates the models and Chroma. The agent is implemented as a Python control loop; the current runtime does not require LangGraph or RAGAS.

The direct route reduces planning overhead; the agent can gather broader evidence at higher cost. Exact-match caching is predictable but does not recognize paraphrases. Context trimming controls token use but may omit useful evidence. These are deliberate trade-offs, not guarantees of answer completeness.

## Evaluation and demonstration

There are two ways to demonstrate the system. For live, reviewer-directed questions, run the interactive Q&A interface:

```powershell
.\.venv\Scripts\python.exe checkpoint_8/interactive_qa.py
```

This loads the corpus and retrieval index, then displays a `You:` prompt. Reviewers can enter their own questions, repeat a question to demonstrate the zero-call cache hit, use `/clear` to clear the session cache, and use `/quit` to exit. Interactive answers and measurements are printed to the terminal but are not saved automatically.

Run the predefined production-readiness evaluation separately:

```powershell
.\.venv\Scripts\python.exe checkpoint_7/capstone_checkpoint_7_1_production_ready_starter.py
```

This makes multiple live API calls. It compares the checkpoint 6 agent with the checkpoint 7 direct route, measures cache miss/hit behavior, exercises a complex question, and runs user-input and poisoned-document injection probes. It appends results to `checkpoint_7_1_agent.log` in the working directory. Logs are excluded from Git by default. The interactive interface prints results but does not save a transcript automatically.

See [the reviewer demonstration guide](DEMO_GUIDE.md) for reproducible questions and observed local evidence. Checkpoint 3 contains the question/grading-note evaluation set; checkpoint 4 compares retrieval approaches; checkpoint 5 evaluates the agent; checkpoint 6 explores security and model costs. Probe outputs require review and do not establish comprehensive security.

## Repository layout

| Path | Contents |
| --- | --- |
| `checkpoint_1/` | Prompt-only baseline. |
| `checkpoint_2/` | Baseline hybrid retrieval. |
| `checkpoint_3/` | Evaluation questions and grading. |
| `checkpoint_4/` | Corpus loading, hybrid/decomposed/graph retrieval. |
| `checkpoint_5/` | Agent retrieval control loop. |
| `checkpoint_6/` | Hardened prompts, usage tracking, security/cost experiments. |
| `checkpoint_7/` | Sanitization, routing, caching, context trimming, evaluation demo. |
| `checkpoint_8/interactive_qa.py` | Interactive entry point. |
| `ResearchPapers/` | PDF corpus. |
| `DEMO_GUIDE.md` | Demonstration procedure and evidence. |

## Limitations and troubleshooting

- This is a local capstone prototype, with no GUI, authentication, hosted service, or multi-user isolation. The checkpoint name "production ready" does not imply a deployed production service.
- Follow-up questions must restate their context. No conversation history is sent to the model.
- Retrieval selects excerpts rather than exhaustively analyzing all papers. Citation rankings such as "most cited paper" are unsupported without citation-count data. Answers may omit explicit paper citations even though source metadata is included in the context.
- PDF extraction can skip content. The observed `Exceeded 5000 form XObject invocations` warning means some form content was omitted; successful startup does not imply complete extraction. Scanned PDFs may require OCR, which this application does not provide.
- Sanitization and XML escaping reduce some injection risks but cannot guarantee protection. Sanitization can remove legitimate wording; the router can misclassify questions. Grounding instructions do not eliminate hallucination.
- Cache entries have no TTL or corpus version. Indexes can also become stale; see the rebuild guidance above.
- API calls send questions and selected paper content to the provider; index creation sends extracted text for embeddings. Use material appropriate for that processing. Latency and token usage vary, and reported chat usage is not a complete monetary cost estimate.
- For `ModuleNotFoundError`, install the root requirements with the same interpreter used to launch the app and keep checkpoints 4–8 together.
- For setup/API failures, check `.env`, model access, account credits, network connectivity, and corpus availability. The CLI displays the exception type rather than raw provider details.
- A fresh install must build its own index unless a compatible one is supplied. Generated indexes and extracted-text caches are intentionally excluded from Git. Ensure reviewers receive the required PDFs or access instructions.
