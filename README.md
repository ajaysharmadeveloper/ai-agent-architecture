# AI Agent Architecture: lesson code

This is the code from my YouTube course, AI Agent Architecture, on [@ajaysharmaAI](https://www.youtube.com/@ajaysharmaAI). Every lesson with code has its own folder under `code/`, and each folder is the exact code I run on screen in that lesson. New lessons land here as they come out.

## Running it

Everything here runs on a local model through [Ollama](https://ollama.com). You don't need an API key, and nothing leaves your machine.

```
ollama pull qwen3:8b
cd code/lesson-16-error-handling-retries-fallbacks
python3 compare.py
```

Most lessons use only the Python standard library. If a lesson needs anything else, its own files say so at the top.

The `*.run.txt` files are the real runs you see in the videos, kept so you can compare your output with mine.

## Lessons

<!-- lessons:start -->
| Lesson | Topic | Code |
|---|---|---|
| 10 | Context Window Budgeting for AI Agents | [code/lesson-10-context-budgeting](code/lesson-10-context-budgeting) |
| 11 | Vector vs Structured Memory for AI Agents | [code/lesson-11-semantic-vs-episodic-memory](code/lesson-11-semantic-vs-episodic-memory) |
| 12 | Agent Checkpointing and Resumability Explained | [code/lesson-12-checkpointing](code/lesson-12-checkpointing) |
| 13 | Tool Schema Design for AI Agents | [code/lesson-13-tool-schema-design](code/lesson-13-tool-schema-design) |
| 14 | Tool Selection and Routing in AI Agents | [code/lesson-14-tool-selection-and-routing](code/lesson-14-tool-selection-and-routing) |
| 15 | Parallel vs Sequential Tool Calls in AI Agents | [code/lesson-15-parallel-tool-calls](code/lesson-15-parallel-tool-calls) |
| 16 | Tool Error Handling and Retries for AI Agents | [code/lesson-16-error-handling-retries-fallbacks](code/lesson-16-error-handling-retries-fallbacks) |
<!-- lessons:end -->

## Stuck?

Ask in the course WhatsApp group (the link is in every video description), or leave a comment on the lesson.
