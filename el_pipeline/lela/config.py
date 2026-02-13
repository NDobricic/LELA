"""Default configuration values for LELA components."""

# NER labels for zero-shot entity recognition
NER_LABELS = ["person", "organization", "location"]

# Model IDs
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"
# Qwen3-4B for entity disambiguation
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"

# Available LLM models for disambiguation (model_id, display_name)
AVAILABLE_LLM_MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-14B", "Qwen3-14B"),
]

# Available embedding models (model_id, display_name)
AVAILABLE_EMBEDDING_MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embed-0.6B"),
    ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embed-4B"),
]

# Available cross-encoder models for reranking
AVAILABLE_CROSS_ENCODER_MODELS = [
    ("tomaarsen/Qwen3-Reranker-4B-seq-cls", "Qwen3-Reranker-4B"),
    ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "Qwen3-Reranker-0.6B"),
]

# vLLM reranker models (seq-cls variants, used with vLLM .score() API)
DEFAULT_VLLM_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
AVAILABLE_VLLM_RERANKER_MODELS = [
    ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "Qwen3-Reranker-0.6B"),
    ("tomaarsen/Qwen3-Reranker-4B-seq-cls", "Qwen3-Reranker-4B"),
]

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
