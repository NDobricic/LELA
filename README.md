# LELA

Standalone, swappable NER → candidate generation → rerank → disambiguation pipeline. Uses file-based storage (JSONL for KB and outputs) and optional caching in `.ner_cache/`.

## Install

**Requirements:** Python >=3.10. GPU + CUDA 12.x only required for the `vllm` extra (local LLM disambiguation/reranking).

### With `uv` (recommended)

```bash
uv sync                            # CLI + library only (CPU-friendly)
uv sync --extra ui                 # + Gradio web UI
uv sync --extra vllm               # + local vLLM (needs CUDA)
uv sync --all-extras               # everything
```

Then prefix commands with `uv run` (e.g. `uv run python -m lela.cli ...`) — no need to activate the venv manually.

### With `pip`

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .                   # core
pip install -e ".[ui]"             # + web UI
pip install -e ".[vllm]"           # + local vLLM
pip install -e ".[ui,vllm]"        # both
```

A pinned core-only `requirements.txt` is also provided for environments where `pip install -e .` doesn't fit; install extras separately with `pip install gradio` / `pip install "vllm>=0.19.0"`.

## Quick start

### Web UI (Gradio)
Requires the `ui` extra (see Install above). Launch:
```bash
uv run python app.py        # or: python app.py
```
Open `http://localhost:7860` and configure the pipeline through the UI. See [docs/WEB_APP.md](docs/WEB_APP.md) for details.

### Troubleshooting

If you encounter issues, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for solutions to common problems including:
- PyTorch CUDA mismatch
- vLLM installation failures
- GPU memory issues

### CLI

**Zero-config quickstart** — uses regex NER, fuzzy string-matching against entity titles, and the YAGO 4.5 KB (auto-downloaded on first run). No GPU, no model downloads:
```bash
python -m lela.cli \
  --config config/quickstart.json \
  --input data/test/sample_doc.txt \
  --output outputs.jsonl
```
First run downloads YAGO (a few hundred MB) and builds the candidate index over it. Subsequent runs are fast — the index is cached under `.ner_cache/`.

Works well when mentions are canonical entity titles (e.g. `Albert Einstein` → `yago:Albert_Einstein`). For ambiguous mentions you'll want a reranker + LLM disambiguator — see `config/lela_example.json`.

With `uv`: `uv sync && uv run python -m lela.cli --config config/quickstart.json --input data/test/sample_doc.txt --output outputs.jsonl`

**Custom config:**
1) Prepare a JSONL knowledge base with fields: `id`, `title`, `description` (plus optional metadata).
2) Create a config file, e.g. `config.json`:
```json
{
  "loader": {"name": "pdf"},
  "ner": {"name": "spacy", "params": {"model": "en_core_web_sm"}},
  "candidate_generator": {"name": "bm25"},
  "reranker": {"name": "none"},
  "disambiguator": {"name": "first"},
  "knowledge_base": {"name": "jsonl", "params": {"path": "kb.jsonl"}}
}
```
3) Run:
```bash
python -m lela.cli --config config.json --input docs/file1.pdf docs/file2.pdf --output outputs.jsonl
```

## Python API

`Lela` accepts a JSON-config path or a dict. Each pipeline stage takes a `name` and an optional `params` block.

```python
from lela import Lela

# Choose each component of LELA
config = {
    "loader": {
        "name": "text"  # or: pdf, docx, html, jsonl, json
    },
    "ner": {
        "name": "gliner",  # or: regex, spacy
        "params": {"labels": ["person", "organization", "location"]},
    },
    "candidate_generator": {"name": "bm25"},
    # or: fuzzy, dense, openai_api_dense
    "reranker": {"name": "llama_server"},
    # or: none, cross_encoder, cross_encoder_vllm, embedder_transformers, embedder_vllm, vllm_api_client
    "disambiguator": {
        "name": "vllm",  # or: first, openai_api, transformers
        "params": {"model_name": "Qwen/Qwen3-4B"},
    },
    "knowledge_base": {
        "name": "jsonl",
        "params": {"path": "my_kb.jsonl"},
    },
}
lela = Lela(config)

# Run the pipeline on a document
results = lela.run("docs/file1.txt")
```

You can also load the config from disk:

```python
lela = Lela("config.json")
results = lela.run("docs/file1.txt", "docs/file2.txt")
```

If you omit the `knowledge_base` block entirely, LELA auto-downloads the YAGO 4.5 KB on first run.

## Available components
- Loaders: `text`, `json`, `jsonl`, `pdf`, `docx`, `html`
- NER: `regex`, `spacy`, `gliner`
- Candidate generators: `bm25`, `fuzzy`, `dense`, `openai_api_dense`
- Rerankers: `none`, `cross_encoder`, `cross_encoder_vllm`, `embedder_transformers`, `embedder_vllm`, `vllm_api_client`, `llama_server`
- Disambiguators: `first`, `vllm`, `transformers`, `openai_api`
- Knowledge bases: `jsonl` (YAGO 4.5 auto-downloads when no `knowledge_base` block is set)

## Sample configs
- `config/quickstart.json` — zero-config, CPU-only, YAGO auto-download (see Quick start above).
- `config/lela_example.json` — full pipeline (GLiNER + dense + cross-encoder + vLLM).
- `config/lela_bm25_only.json` — minimal BM25-only setup.
- `config/test_gliner_fuzzy_ce_transformers.json` — uses `transformers` disambiguator (no vLLM).

## Conversion utilities
- YAGO labels TSV → JSONL KB:
  ```bash
  python -m lela.scripts.convert_yago_labels data/kb/yagoLabels.tsv data/kb/yago_labels_en.jsonl
  ```

## Notes
- Outputs are JSONL (one line per document with resolved entities).
  - Each line: `id`, `text`, `entities` (with `text`, `start`, `end`, `label`, `entity_id`, `entity_title`, `entity_description`, `candidates`).
- Cache lives in `.ner_cache/` keyed by file path, mtime, and size.
- No dependency on LELA; integration would be optional if added later.
