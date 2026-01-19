"""Default configuration values for LELA components."""

# NER labels for zero-shot entity recognition
NER_LABELS = ["person", "organization", "location", "event", "work of art", "product"]

# Model IDs
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"

# Retrieval settings
CANDIDATES_TOP_K = 64
RERANKER_TOP_K = 10

# Span markers for disambiguation prompts
SPAN_OPEN = "["
SPAN_CLOSE = "]"

# Special entity label
NOT_AN_ENTITY = "None"

# vLLM settings
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = None

# Embedding task descriptions
RETRIEVER_TASK = (
    "Given an ambiguous mention, retrieve relevant entities that the mention refers to."
)
RERANKER_TASK = (
    "Given a text with a marked mention enclosed in square brackets, "
    "retrieve relevant entities that the mention refers to."
)

# Default generation config for vLLM
DEFAULT_GENERATION_CONFIG = {
    "max_tokens": 32768,
}
