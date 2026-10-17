# music-llm
A music transformer trained on MIDI tokens to (try to) generate musical sequences

## Setup

```bash
uv sync
uv run python -m src.utils.train --help
```

See `AGENTS.md` for commands and layout, and `docs/` for the architecture, data pipeline,
scaling notes and Colab workflow.
