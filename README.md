<div align="center">

# LELA

**A modular, end-to-end entity-linking pipeline.**

*Find entities in text → match them to a knowledge base → swap any stage with one line of JSON.*

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org)
[![Conference](https://img.shields.io/badge/IJCAI--ECAI-2026_Demo-orange)](https://arxiv.org/abs/2605.26956)
[![YAGO Ecosystem](https://img.shields.io/badge/part_of-YAGO_ecosystem-yellow)](https://yago-knowledge.org)

</div>

---

## Why LELA

Entity linking — finding and mapping mentions in text to their corresponding entities in a Knowledge Base (KB) — usually means stitching together different tools that are often limited to linking to Wikipedia. **LELA replaces that with a single config file.** Five swappable stages (loader → NER → candidate generation → reranking → disambiguation) plus a pluggable knowledge base, all wired into one Python class or one CLI call.

```text
  ┌────────┐   ┌──────┐   ┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
  │  text  │ → │ NER  │ → │ candidates │ → │ reranker │ → │ disambiguator│ → │ entities │
  └────────┘   └──────┘   └────────────┘   └──────────┘   └──────────────┘   └──────────┘
                              ▲                                   ▲
                              └────── KB (Custom/YAGO 4.5) ───────┘
```

**Highlights**

- **Zero-config quickstart** — `git clone && uv sync && uv run python -m lela.cli ...` works on CPU with no model downloads. YAGO 4.5 fetches itself on first use.
- **Bring your own KB** — any JSONL file with `id`, `title`, `description` plugs straight in.
- **Mix and match** — regex/spaCy/GLiNER for NER, BM25/fuzzy/dense for candidates, cross-encoder/embedder rerankers, and vLLM / Hugging Face Transformers / OpenAI-compatible API disambiguators.
- **Two interfaces** — Python API for embedding into your workflows, a Gradio web UI for hands-on exploration.
- **CPU-friendly defaults, GPU when you need it** — vLLM is an optional extra; everything else runs on a laptop.

---

## Quickstart

```bash
git clone https://github.com/<your-org>/lela.git
cd lela
uv sync
uv run python -m lela.cli \
  --config config/quickstart.json \
  --input data/test/sample_doc.txt \
  --output outputs.jsonl
```

This runs on CPU with **no model downloads**. The first invocation fetches YAGO 4.5 (a few hundred MB; one-time, cached under `.ner_cache/`). On the sample document `"Albert Einstein was born in Germany. Marie Curie was a pioneering scientist."` you should see:

```jsonl
{"text": "Albert Einstein", "entity_id": "yago:Albert_Einstein", ...}
{"text": "Germany",         "entity_id": "yago:Germany",         ...}
{"text": "Marie Curie",     "entity_id": "yago:Marie_Curie",     ...}
```

For ambiguous mentions you'll want a heavier config — see the [recommended configurations](#recommended-configurations) below.

---

## Install

**Requirements:** Python ≥3.10. GPU + CUDA 12.x only required for the `vllm` extra (local LLM disambiguation/reranking).

**Platform support:**
- **Linux** — fully supported, including the `vllm` extra.
- **macOS** — core + `ui` extra supported. `vllm` is not available; use `openai_api` disambiguator pointing at a remote server (or the `transformers` disambiguator for small models on CPU).
- **Windows** — not officially tested; WSL2 is the recommended workaround.

First, clone the repo:

```bash
git clone https://github.com/<your-org>/lela.git
cd lela
```

### With `uv` (recommended on Linux/macOS)

```bash
uv sync                            # CLI + library only (CPU-friendly)
uv sync --extra ui                 # + Gradio web UI
uv sync --extra vllm               # + local vLLM (needs CUDA)
uv sync --all-extras               # everything
```

Then prefix commands with `uv run` (e.g. `uv run python -m lela.cli ...`) — no need to activate the venv manually.

> **Windows users:** `uv` workflows aren't tested on native Windows. Use the `pip` path below from a regular PowerShell / Command Prompt, or run everything inside WSL2 (where `uv` works as on Linux).

### With `pip`

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows (PowerShell / cmd)

python -m pip install --upgrade pip
python -m pip install -e .                  # core
python -m pip install -e ".[ui]"            # + web UI
python -m pip install -e ".[vllm]"          # + local vLLM (Linux only)
python -m pip install -e ".[ui,vllm]"       # both
```

A pinned core-only `requirements.txt` is also provided for environments where `pip install -e .` doesn't fit; install extras separately with `python -m pip install gradio` / `python -m pip install "vllm>=0.19.0"`.

---

## Recommended configurations

Pick a row that matches your hardware and quality target. All four configs live in `config/` and can be used with the CLI directly (e.g. `python -m lela.cli --config config/quickstart.json --input docs/file1.txt`).

| Use case | NER | Candidates | Reranker | Disambiguator | Hardware | Config |
|---|---|---|---|---|---|---|
| **Fast / instant demo** | `regex` | `fuzzy` | none | `first` | CPU only | [`config/quickstart.json`](config/quickstart.json) |
| **Better NER, still CPU** | `gliner` | `bm25` | none | `first` | CPU | [`config/lela_bm25_only.json`](config/lela_bm25_only.json) |
| **Strong, no LLM** | `gliner` | `dense` (0.6B) | `cross_encoder` (0.6B) | `first` | CPU works; 1× GPU much faster | [`config/lela_strong_cpu.json`](config/lela_strong_cpu.json) |
| **Strong + LLM via llama.cpp** | `gliner` | `dense` (0.6B) | `cross_encoder` (0.6B) | `openai_api` → `llama-server` | CPU only (quantized model) | [`config/lela_strong_llamacpp.json`](config/lela_strong_llamacpp.json) |
| **Best quality** | `gliner` | `dense` (4B, +context) | `cross_encoder` (4B) | `vllm` (Qwen3-4B) | 1× GPU (~24+ GB) | [`config/lela_example.json`](config/lela_example.json) |
| **API-only (no local GPU)** | `gliner` | `bm25` | none | `openai_api` | CPU + remote LLM | build your own — see [`docs/API.md`](docs/API.md) |

> **Running with llama.cpp:** the `lela_strong_llamacpp.json` config expects `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) to be running locally on port 8080. Start it before LELA, e.g.:
> ```bash
> llama-server -m models/Qwen3-4B-Instruct-Q4_K_M.gguf -c 8192 --port 8080
> ```
> The same config also works against any other OpenAI-compatible endpoint (Ollama, vLLM-as-server, Together, etc.) — just edit `base_url` and `model_name`.

Rough quality / cost trade-off:
- `regex + fuzzy + first` works perfectly when mentions are canonical entity titles (e.g. "Albert Einstein"), and fails on ambiguous mentions.
- Adding `gliner` improves NER quality on noisy/typed text and supports custom entity labels.
- Adding a `dense` or `cross_encoder` reranker is the biggest quality jump when the KB is large (BM25/fuzzy top-1 isn't great by itself).
- An LLM disambiguator (`vllm`, `transformers`, or `openai_api`) handles ambiguity from context through LLM-based reasoning — but costs the most.

---

## Usage

### CLI

```bash
python -m lela.cli --config config.json --input docs/file1.pdf docs/file2.pdf --output outputs.jsonl
```

Inputs can be `txt`, `pdf`, `docx`, `html`, `json`, or `jsonl`. Output is one JSONL document per input file with resolved entities, candidates, and metadata. See [`docs/CLI.md`](docs/CLI.md) for the full reference.

### Python API

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

Omit the `knowledge_base` block entirely and LELA auto-downloads YAGO 4.5 on first run.

### Web UI

Requires the `ui` extra (see [Install](#install)):

```bash
uv run python app.py        # or: python app.py
```

Open `http://localhost:7860` and configure the pipeline through the UI. See [`docs/WEB_APP.md`](docs/WEB_APP.md) for details.

---

## Available components

- **Loaders:** `text`, `json`, `jsonl`, `pdf`, `docx`, `html`
- **NER:** `regex`, `spacy`, `gliner`
- **Candidate generators:** `bm25`, `fuzzy`, `dense`, `openai_api_dense`
- **Rerankers:** `none`, `cross_encoder`, `cross_encoder_vllm`, `embedder_transformers`, `embedder_vllm`, `vllm_api_client`, `llama_server`
- **Disambiguators:** `first`, `vllm`, `transformers`, `openai_api`
- **Knowledge bases:** `jsonl` (custom KB), `yago` (auto-downloads YAGO 4.5). Omitting the `knowledge_base` block entirely is equivalent to `"name": "yago"`.

Full per-component reference: [`docs/PIPELINE.md`](docs/PIPELINE.md) · [`docs/API.md`](docs/API.md)

---

## Output format

Each line of the output JSONL contains one document:

```json
{
  "id": "sample_doc",
  "text": "Albert Einstein was born in Germany. ...",
  "entities": [
    {
      "text": "Albert Einstein",
      "start": 0, "end": 15,
      "label": "ENT",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "yago:Albert_Einstein",
      "entity_title": "Albert_Einstein",
      "entity_description": "...",
      "candidates": [{"entity_id": "...", "score": 1.0, "description": "..."}, ...]
    }
  ],
  "meta": {"source": "data/test/sample_doc.txt"}
}
```

Cache is keyed by file path, mtime, and size, and lives in `.ner_cache/`.

---

## Conversion utilities

- YAGO labels TSV → JSONL KB:
  ```bash
  python -m lela.scripts.convert_yago_labels data/kb/yagoLabels.tsv data/kb/yago_labels_en.jsonl
  ```

---

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — component architecture and the spaCy integration.
- [`docs/API.md`](docs/API.md) — Python API and component config reference.
- [`docs/CLI.md`](docs/CLI.md) — command-line reference and example configs.
- [`docs/WEB_APP.md`](docs/WEB_APP.md) — Gradio web UI.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — installation and runtime issues.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — hardware sizing.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — contributing.

---

## Citation

If you use LELA in your research, please cite:

```bibtex
@inproceedings{lela2026,
  title     = {LELA: An End-to-End LLM-based Entity Linking Framework with Zero-shot Domain Aadaptation},
  author    = {Samy Haffoudhi , Nikola Dobričić , Fabian Suchanek , Nils Holzenberger},
  booktitle = {35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.26956}
}
```

## Authors

- [Samy Haffoudhi](https://samyhaff.github.io/)
- [Nikola Dobričić](https://github.com/NDobricic)
- [Fabian Suchanek](https://suchanek.name/)
- [Nils Holzenberger](https://perso.telecom-paristech.fr/holzenberger/)

## Acknowledgements

LELA is part of the [YAGO knowledge graph ecosystem](https://yago-knowledge.org).

## License

LELA is licensed under the [Apache License 2.0](LICENSE).
