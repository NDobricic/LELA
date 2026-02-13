"""Default configuration values for LELA components."""

# NER labels for zero-shot entity recognition
NER_LABELS = ["person", "organization", "location"]

# Model IDs
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"
DEFAULT_GLINER_VRAM_GB = 1.0  # NuNER_Zero-span is a small model
# Qwen3-4B for entity disambiguation
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"

# Available LLM models for disambiguation (model_id, display_name, vram_gb)
# VRAM estimates: params * 2 bytes (fp16) + ~15% overhead
AVAILABLE_LLM_MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", 1.5),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", 4.0),
    ("Qwen/Qwen3-4B", "Qwen3-4B", 9.5),
    ("Qwen/Qwen3-8B", "Qwen3-8B", 18.5),
    ("Qwen/Qwen3-14B", "Qwen3-14B", 32.5),
]

# Available embedding models (model_id, display_name, vram_gb)
AVAILABLE_EMBEDDING_MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embed-0.6B", 1.5),
    ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embed-4B", 9.5),
]

# Available cross-encoder models for reranking (model_id, display_name, vram_gb)
AVAILABLE_CROSS_ENCODER_MODELS = [
    ("tomaarsen/Qwen3-Reranker-4B-seq-cls", "Qwen3-Reranker-4B", 9.5),
    ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "Qwen3-Reranker-0.6B", 1.5),
]

# vLLM reranker models (seq-cls variants, used with vLLM .score() API)
DEFAULT_VLLM_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
AVAILABLE_VLLM_RERANKER_MODELS = [
    ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "Qwen3-Reranker-0.6B", 1.5),
    ("tomaarsen/Qwen3-Reranker-4B-seq-cls", "Qwen3-Reranker-4B", 9.5),
]


def get_model_vram_gb(model_id: str) -> float:
    """Look up known VRAM usage (in GB) for a model. Returns 2.0 as fallback."""
    for model_list in (
        AVAILABLE_LLM_MODELS,
        AVAILABLE_EMBEDDING_MODELS,
        AVAILABLE_CROSS_ENCODER_MODELS,
        AVAILABLE_VLLM_RERANKER_MODELS,
    ):
        for entry in model_list:
            if entry[0] == model_id:
                return entry[2]
    return 2.0

# Qwen3-Reranker prompt templates (no colons after tags)
CROSS_ENCODER_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
CROSS_ENCODER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CROSS_ENCODER_QUERY_TEMPLATE = "{prefix}<Instruct>{instruction}\n<Query>{query}{suffix}"
CROSS_ENCODER_DOCUMENT_TEMPLATE = "<Document>{doc}{suffix}"

# Retrieval settings
CANDIDATES_TOP_K = 64
RERANKER_TOP_K = 10

# Span markers for disambiguation prompts
SPAN_OPEN = "["
SPAN_CLOSE = "]"

# Special entity label
NOT_AN_ENTITY = ""

# vLLM settings
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = 8192
VLLM_GPU_MEMORY_UTILIZATION = 0.9  # Default fraction of GPU memory vLLM will use

# Embedding task descriptions
RETRIEVER_TASK = (
    "Given an ambiguous mention, retrieve relevant entities that the mention refers to."
)
RERANKER_TASK = (
    "Given a text with a mention enclosed between the '[' and ']' characters, "
    "retrieve relevant entities that the mention refers to."
)

# Default generation config for LLM disambiguation
# Qwen3 needs more tokens due to thinking mode
DEFAULT_GENERATION_CONFIG = {
    "max_tokens": 4096,  # More tokens for thinking mode + answer
    # "temperature": 0.1,  # Low temperature for deterministic outputs
    # "top_p": 0.9,
    # "repetition_penalty": 1.1,  # Prevent repetitive garbage output
}
