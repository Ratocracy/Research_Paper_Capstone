# Reviewer demonstration and evidence

Follow the root README to install dependencies and configure `OPENROUTER_API_KEY` before using this guide. Live runs require internet access, can incur API charges, and should be started from the repository root. Use the full research-paper collection for results comparable to the observations below.

The project provides two demonstrations: interactive Q&A for reviewer-directed questions and a predefined production-readiness evaluation for repeatable engineering evidence.

## Interactive Q&A demonstration

Start the interactive interface on Windows:

```powershell
.\.venv\Scripts\python.exe checkpoint_8/interactive_qa.py
```

On macOS/Linux, run:

```bash
.venv/bin/python checkpoint_8/interactive_qa.py
```

Wait for `Ready: ... paper chunks.` and the `You:` prompt, then follow this procedure:

1. Wait for the corpus to load and the `You:` prompt to appear.
2. Enter `how does sketching work in program synthesis?`. Review the explanation against the papers. Expect the direct route.
3. Enter the identical question again. Expect the same answer, `cache-hit`, zero model calls, and zero model tokens.
4. Enter `/clear` and repeat it. Expect a fresh model call.
5. Enter `Compare the major approaches to program synthesis discussed across the papers.`. Expect the agent route, printed retrieval steps, and a final answer. The selected actions and answer vary across runs.
6. Enter `what is the most cited paper and why is it the most cited`. Assess whether the answer acknowledges the lack of citation-count evidence rather than inventing a ranking. This is a limitation probe, not a supported citation-analysis feature.
7. Enter `/quit`.

The interface prints each answer together with its route, tracked chat-model calls, tracked chat-model tokens, and elapsed time. It does not save a transcript automatically. `/clear` resets only the in-memory answer cache; it does not delete the retrieval index.

## Production-readiness evaluation

Run the predefined production-readiness evaluation separately:

```powershell
.\.venv\Scripts\python.exe checkpoint_7/capstone_checkpoint_7_1_production_ready_starter.py
```

On macOS/Linux, run:

```bash
.venv/bin/python checkpoint_7/capstone_checkpoint_7_1_production_ready_starter.py
```

This makes multiple live API calls. It compares the checkpoint 6 agent with the checkpoint 7 direct route, measures cache miss/hit behavior, exercises a complex question, and runs user-input and poisoned-document injection probes. It appends results to `checkpoint_7_1_agent.log` in the working directory. Logs are excluded from Git by default. The interactive interface prints results but does not save a transcript automatically.

Review the evaluation output for:

- the checkpoint 6 agent versus checkpoint 7 direct-route call, token, and runtime measurements;
- the cache miss versus identical cache-hit measurements;
- the complex-question agent trace and final answer;
- raw versus sanitized user-injection behavior; and
- the poisoned retrieved-document probe.

Probe outputs require human review and do not establish comprehensive security.

## Observed local evidence

These observations were supplied by the project author during interactive testing. They are illustrative measurements, not averages or independent quality scores. The startup loaded 153 PDFs into 5,323 chunks and reported a PDF form-extraction limit warning.

| Question/run | Route | Chat model calls | Chat model tokens | Elapsed time |
| --- | --- | --- | --- | --- |
| `how does sketching work in program synthesis?` | direct | 1 | 4,932 | 2.67 s |
| Identical repeated question | cache-hit | 0 | 0 | 0.00 s (rounded) |
| `what is the most cited paper and why is it the most cited` | direct | 1 | 6,659 | 2.35 s |

The sketching response described partial programs, holes, specifications, and automated completion. The citation-ranking response stated that the retrieved documents did not establish a single most-cited paper. These outputs demonstrate behavior; their full factual accuracy was not independently scored here. The repeated query demonstrates avoided chat calls, not a measurement of overall system operating cost.

## Broader evaluation context

Preserve selected outputs for review, including the question, route, answer, usage, and runtime. Avoid including credentials in shared evidence. Checkpoint 3 contains the question/grading-note evaluation set; checkpoint 4 compares retrieval approaches; checkpoint 5 evaluates the agent; and checkpoint 6 explores security and model costs. No new benchmark scores are claimed by this document.

## Setup verification

On October 13, 2026, the root requirements installed successfully into a separate Windows Python 3.11.13 virtual environment. `python -m pip check` reported no broken requirements. The checkpoint 8 import chain loaded successfully, and a smoke check exercised startup/quit with mocked corpus loading and API configuration, plus direct/agent routing. Documentation links and the presence of 153 local PDFs were also checked.

This validates dependency installation and an offline startup path, not a fresh-clone live indexing/answer run. No paid API calls were made during that check. The live observations above came from the author's existing environment. macOS/Linux installation was not tested. The requirements pin direct dependencies but are not a complete transitive lockfile.
