# docs

Index of everything under `docs/`. Start with `PLAN.md` for the work in progress.

## Planning

| File | What it holds |
|---|---|
| [PLAN.md](PLAN.md) | The current story: the roadmap phase in progress, broken into tasks with acceptance criteria |
| [ROADMAP.md](ROADMAP.md) | Long-term phases (workstreams), their goals and exit criteria |
| [known-issues.md](known-issues.md) | Numbered bugs and gaps with file:line. Tasks in `PLAN.md` refer to them by number. |

## Architecture

How the system works today. Update these when the code changes.

| File | What it holds |
|---|---|
| [architecture/model-and-training.md](architecture/model-and-training.md) | Model structure, relative attention, the parameter-count formula, training loop, checkpoints, inference, change history |
| [architecture/data-pipeline.md](architecture/data-pipeline.md) | Extraction, tokenization, augmentation, splits, datasets, corpus sizes, how to rebuild the data |
| [architecture/scaling.md](architecture/scaling.md) | Current model size, run history, the parameters-vs-data analysis |

## Guides

| File | What it holds |
|---|---|
| [colab.md](colab.md) | Running training on Colab GPUs with the `colab` CLI (not yet tested end to end) |

## Research and history

| File | What it holds |
|---|---|
| [references.md](references.md) | Papers, datasets, repos and docs the project uses or plans to use |
| [journal/](journal/README.md) | Dated project log from Apr 2025 onward (migrated from Notion), plus learning notes |

The long-form research write-up "Scaling music-llm" (Oct 2026) is a private artifact:
https://claude.ai/artifact/2DwzEbqu9RBaqa1uVAorge. `ROADMAP.md` is its in-repo summary.
