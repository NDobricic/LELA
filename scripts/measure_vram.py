#!/usr/bin/env python3
"""Measure actual VRAM usage for each model in the pipeline.

Loads each model one by one, measures GPU memory before/after,
and prints a table of model → actual VRAM usage.

Usage:
    python scripts/measure_vram.py
    python scripts/measure_vram.py --models Qwen/Qwen3-4B Qwen/Qwen3-Embedding-4B
"""

import argparse
import gc
import sys

import torch

from lela.lela.config import (
    AVAILABLE_LLM_MODELS,
    AVAILABLE_EMBEDDING_MODELS,
    AVAILABLE_CROSS_ENCODER_MODELS,
    DEFAULT_GLINER_MODEL,
)


def get_free_vram() -> float:
    """Get free GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(0)
    return free / (1024**3)


def get_used_vram() -> float:
    """Get used GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / (1024**3)


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_sentence_transformer(model_name: str) -> float:
    """Measure VRAM for a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    clear_gpu()
    before = get_used_vram()

    model = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": torch.float16},
        trust_remote_code=True,
    )

    torch.cuda.synchronize()
    after = get_used_vram()

    del model
    clear_gpu()

    return after - before


def measure_causal_lm(model_name: str) -> float:
    """Measure VRAM for a CausalLM model (transformers)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    clear_gpu()
    before = get_used_vram()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    torch.cuda.synchronize()
    after = get_used_vram()

    del model, tokenizer
    clear_gpu()

    return after - before


def measure_gliner(model_name: str) -> float:
    """Measure VRAM for a GLiNER model."""
    from gliner import GLiNER

    clear_gpu()
    before = get_used_vram()

    model = GLiNER.from_pretrained(model_name)

    torch.cuda.synchronize()
    after = get_used_vram()

    del model
    clear_gpu()

    return after - before


def measure_cross_encoder(model_name: str) -> float:
    """Measure VRAM for a CrossEncoder model."""
    from sentence_transformers import CrossEncoder

    clear_gpu()
    before = get_used_vram()

    model = CrossEncoder(model_name)

    torch.cuda.synchronize()
    after = get_used_vram()

    del model
    clear_gpu()

    return after - before


def main():
    parser = argparse.ArgumentParser(description="Measure VRAM usage per model")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model IDs to measure (default: all known models)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {total:.1f} GB")
    print()

    # Build list of models to measure
    models_to_measure = []

    if args.models:
        for model_id in args.models:
            # Try to auto-detect the type
            if "Embedding" in model_id:
                models_to_measure.append((model_id, "embedding"))
            elif "Reranker" in model_id:
                models_to_measure.append((model_id, "cross_encoder"))
            elif "GLiNER" in model_id or "NuNER" in model_id:
                models_to_measure.append((model_id, "gliner"))
            else:
                models_to_measure.append((model_id, "causal_lm"))
    else:
        # Measure all known models
        models_to_measure.append((DEFAULT_GLINER_MODEL, "gliner"))
        for model_id, _, _ in AVAILABLE_EMBEDDING_MODELS:
            models_to_measure.append((model_id, "embedding"))
        for model_id, _, _ in AVAILABLE_CROSS_ENCODER_MODELS:
            models_to_measure.append((model_id, "cross_encoder"))
        for model_id, _, _ in AVAILABLE_LLM_MODELS:
            models_to_measure.append((model_id, "causal_lm"))

    # Measure each model
    results = []
    for model_id, model_type in models_to_measure:
        print(f"Measuring {model_id} ({model_type})...", end=" ", flush=True)
        try:
            if model_type == "embedding":
                vram = measure_sentence_transformer(model_id)
            elif model_type == "causal_lm":
                vram = measure_causal_lm(model_id)
            elif model_type == "gliner":
                vram = measure_gliner(model_id)
            elif model_type == "cross_encoder":
                vram = measure_cross_encoder(model_id)
            else:
                print("SKIP (unknown type)")
                continue
            print(f"{vram:.2f} GB")
            results.append((model_id, model_type, vram))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((model_id, model_type, None))

    # Print summary table
    print()
    print("=" * 70)
    print(f"{'Model':<45} {'Type':<15} {'VRAM (GB)':>10}")
    print("-" * 70)
    for model_id, model_type, vram in results:
        vram_str = f"{vram:.2f}" if vram is not None else "FAILED"
        print(f"{model_id:<45} {model_type:<15} {vram_str:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
