# AI Agent Architecture: lesson code

This is the code from my YouTube course, AI Agent Architecture, on [@ajaysharmaAI](https://www.youtube.com/@ajaysharmaAI). Every lesson has its own folder under `code/`, and each folder is the exact code I run on screen in that lesson.

## Running it

From Lesson 10 on, everything runs on a local model through [Ollama](https://ollama.com). You don't need an API key, and nothing leaves your machine.

```
ollama pull qwen3:8b
cd code/lesson-16-error-handling-retries-fallbacks
python3 compare.py
```

Most lessons use only the Python standard library. If a lesson needs anything else, its own files say so at the top. Lessons 7 to 9 came before I switched to a local model, so they call a hosted API and read the key from your environment.

The `*.run.txt` files are the real runs you see in the videos, kept so you can compare your output with mine.

## Lessons

<!-- lessons:start -->
<!-- lessons:end -->

Lessons 1 to 6 were concept lessons with no code.

## Stuck?

Ask in the course WhatsApp group (the link is in every video description), or leave a comment on the lesson.
