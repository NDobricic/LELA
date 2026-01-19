import os
# Disable vLLM V1 engine before any imports - V1 uses multiprocessing that fails from worker threads
os.environ["VLLM_USE_V1"] = "0"

import argparse
import importlib.util
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.registry import (
    candidate_generators,
    disambiguators,
    knowledge_bases,
    loaders,
    ner_models,
    rerankers,
)

DESCRIPTION = """
# NER Pipeline 🔗

Modular NER → candidate generation → rerank → disambiguation pipeline. 
Swap components, configure parameters, and test with your own knowledge bases.
"""


def _is_vllm_usable() -> bool:
    """Check if vllm is installed and CUDA is available for it to run."""
    if importlib.util.find_spec("vllm") is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_available_components() -> Dict[str, List[str]]:
    """Get list of available components from registries."""
    available_disambiguators = list(disambiguators.available().keys())
    if not _is_vllm_usable():
        available_disambiguators = [d for d in available_disambiguators if d != "lela_vllm"]
    
    return {
        "loaders": list(loaders.available().keys()),
        "ner": list(ner_models.available().keys()),
        "candidates": list(candidate_generators.available().keys()),
        "rerankers": ["none"] + list(rerankers.available().keys()),
        "disambiguators": ["none"] + available_disambiguators,
        "knowledge_bases": list(knowledge_bases.available().keys()),
    }


def format_highlighted_text(result: Dict) -> List[Tuple[str, Optional[str]]]:
    """Convert pipeline result to HighlightedText format."""
    text = result["text"]
    entities = result["entities"]
    
    if not entities:
        return [(text, None)]
    
    sorted_entities = sorted(entities, key=lambda e: e["start"])
    
    highlighted = []
    last_end = 0
    
    for entity in sorted_entities:
        if entity["start"] > last_end:
            highlighted.append((text[last_end:entity["start"]], None))
        
        label = entity.get("label", "ENT")
        if entity.get("entity_title"):
            label = f"{label}: {entity['entity_title']}"
        
        highlighted.append((entity["text"], label))
        last_end = entity["end"]
    
    if last_end < len(text):
        highlighted.append((text[last_end:], None))
    
    return highlighted


def run_pipeline(
    text_input: str,
    file_input: Optional[gr.File],
    kb_file: Optional[gr.File],
    loader_type: str,
    ner_type: str,
    spacy_model: str,
    gliner_model: str,
    gliner_labels: str,
    simple_min_len: int,
    cand_type: str,
    cand_top_k: int,
    reranker_type: str,
    disambig_type: str,
    progress=gr.Progress(),
) -> Tuple[List[Tuple[str, Optional[str]]], Dict]:
    """Run the NER pipeline with selected configuration."""
    
    if not kb_file:
        raise gr.Error("Please upload a knowledge base JSONL file.")
    
    if not text_input and not file_input:
        raise gr.Error("Please provide either text input or upload a file.")
    
    progress(0.1, desc="Building pipeline configuration...")
    
    ner_params = {}
    if ner_type == "spacy":
        ner_params["model"] = spacy_model
    elif ner_type == "gliner":
        ner_params["model_name"] = gliner_model
        if gliner_labels:
            ner_params["labels"] = [l.strip() for l in gliner_labels.split(",")]
    elif ner_type == "simple":
        ner_params["min_len"] = simple_min_len
    
    cand_params = {}
    if cand_top_k > 0:
        cand_params["top_k"] = cand_top_k
    
    config_dict = {
        "loader": {"name": loader_type, "params": {}},
        "ner": {"name": ner_type, "params": ner_params},
        "candidate_generator": {"name": cand_type, "params": cand_params},
        "reranker": {"name": reranker_type, "params": {}} if reranker_type != "none" else None,
        "disambiguator": {"name": disambig_type, "params": {}} if disambig_type != "none" else None,
        "knowledge_base": {"name": "custom", "params": {"path": kb_file.name}},
        "cache_dir": ".ner_cache",
        "batch_size": 1,
    }
    
    progress(0.3, desc="Initializing pipeline...")
    
    try:
        config = PipelineConfig.from_dict(config_dict)
        pipeline = NERPipeline(config)
    except Exception as e:
        raise gr.Error(f"Failed to initialize pipeline: {str(e)}")
    
    progress(0.5, desc="Running pipeline...")
    
    try:
        if file_input:
            input_path = file_input.name
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(text_input)
                input_path = f.name
        
        results = pipeline.run([input_path])
        
        if not file_input:
            os.unlink(input_path)
        
        if not results:
            raise gr.Error("No results returned from pipeline.")
        
        result = results[0]
        
    except Exception as e:
        raise gr.Error(f"Pipeline execution failed: {str(e)}")
    
    progress(0.9, desc="Formatting output...")
    
    highlighted = format_highlighted_text(result)
    
    progress(1.0, desc="Done!")
    
    return highlighted, result


def update_ner_params(ner_choice: str):
    """Show/hide NER-specific parameters based on selection."""
    return {
        spacy_params: gr.update(visible=(ner_choice == "spacy")),
        gliner_params: gr.update(visible=(ner_choice == "gliner")),
        simple_params: gr.update(visible=(ner_choice == "simple")),
    }


def update_loader_from_file(file: Optional[gr.File]):
    """Auto-detect loader type from file extension."""
    if not file:
        return gr.update()
    
    ext = Path(file.name).suffix.lower()
    loader_map = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
    }
    
    if ext in loader_map:
        return gr.update(value=loader_map[ext])
    return gr.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Pipeline Gradio UI")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log)
    logger = logging.getLogger(__name__)
    
    components = get_available_components()
    
    with gr.Blocks(title="NER Pipeline", fill_height=True) as demo:
        gr.Markdown(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter text to process...",
                    lines=8,
                    value="Albert Einstein was born in Germany. Marie Curie was a pioneering scientist.",
                )
                
                file_input = gr.File(
                    label="Or Upload Document",
                    file_types=[".txt", ".pdf", ".docx", ".html"],
                )
                
                kb_file = gr.File(
                    label="Knowledge Base (JSONL)",
                    file_types=[".jsonl"],
                    value="data/test/sample_kb.jsonl" if os.path.exists("data/test/sample_kb.jsonl") else None,
                )
                
                gr.Markdown("### Pipeline Configuration")
                
                with gr.Accordion("Loader", open=False):
                    loader_type = gr.Dropdown(
                        choices=components["loaders"],
                        value="text",
                        label="Loader Type",
                        info="Auto-detected from file extension",
                    )
                
                with gr.Accordion("Named Entity Recognition (NER)", open=True):
                    ner_type = gr.Dropdown(
                        choices=components["ner"],
                        value="simple",
                        label="NER Model",
                    )
                    
                    with gr.Group(visible=False) as spacy_params:
                        spacy_model = gr.Textbox(
                            label="SpaCy Model",
                            value="en_core_web_sm",
                            info="Requires: python -m spacy download en_core_web_sm",
                        )
                    
                    with gr.Group(visible=False) as gliner_params:
                        gliner_model = gr.Textbox(
                            label="GLiNER Model",
                            value="urchade/gliner_large",
                        )
                        gliner_labels = gr.Textbox(
                            label="Entity Labels (comma-separated)",
                            value="person, organization, location",
                        )
                    
                    with gr.Group(visible=True) as simple_params:
                        simple_min_len = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Minimum Length",
                        )
                
                with gr.Accordion("Candidate Generation", open=True):
                    cand_type = gr.Dropdown(
                        choices=components["candidates"],
                        value="fuzzy",
                        label="Candidate Generator",
                    )
                    cand_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Top K Candidates",
                    )
                
                with gr.Accordion("Reranking", open=False):
                    reranker_type = gr.Dropdown(
                        choices=components["rerankers"],
                        value="none",
                        label="Reranker",
                    )
                
                with gr.Accordion("Disambiguation", open=True):
                    disambig_type = gr.Dropdown(
                        choices=components["disambiguators"],
                        value="first",
                        label="Disambiguator",
                    )
                
                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                highlighted_output = gr.HighlightedText(
                    label="Linked Entities",
                    color_map={},
                    show_legend=True,
                )
                
                json_output = gr.JSON(
                    label="Full Pipeline Output",
                )
        
        file_input.change(
            fn=update_loader_from_file,
            inputs=[file_input],
            outputs=[loader_type],
        )
        
        ner_type.change(
            fn=update_ner_params,
            inputs=[ner_type],
            outputs=[spacy_params, gliner_params, simple_params],
        )
        
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                text_input,
                file_input,
                kb_file,
                loader_type,
                ner_type,
                spacy_model,
                gliner_model,
                gliner_labels,
                simple_min_len,
                cand_type,
                cand_top_k,
                reranker_type,
                disambig_type,
            ],
            outputs=[highlighted_output, json_output],
        )
        
        gr.Markdown("""
## Quick Start

1. **Upload Knowledge Base**: Provide a JSONL file with entities (required fields: `id`, `title`, `description`)
2. **Enter Text or Upload File**: Input text directly or upload a document
3. **Configure Pipeline**: Select components and adjust parameters
4. **Run**: Click "Run Pipeline" to process

### Example Files

Test files are available in `data/test/`:
- `sample_kb.jsonl` - Sample knowledge base with 10 entities
- `sample_doc.txt` - Sample document for testing
        """)
        
        with gr.Accordion("Entity Type Legend (SpaCy)", open=False):
            gr.Markdown("""
| Label | Meaning | Example |
|-------|---------|---------|
| **PERSON** | People, including fictional | *Albert Einstein*, *Marie Curie* |
| **ORG** | Organizations, companies, agencies | *Google*, *United Nations*, *NASA* |
| **GPE** | Geopolitical entities (countries, cities, states) | *France*, *New York*, *California* |
| **LOC** | Non-GPE locations (mountains, water bodies) | *Mount Everest*, *Pacific Ocean* |
| **FAC** | Facilities (buildings, airports, highways) | *Empire State Building*, *JFK Airport* |
| **PRODUCT** | Objects, vehicles, foods (not services) | *iPhone*, *Boeing 747* |
| **EVENT** | Named events (hurricanes, battles, wars) | *World War II*, *Hurricane Katrina* |
| **WORK_OF_ART** | Titles of books, songs, etc. | *The Great Gatsby*, *Mona Lisa* |
| **LAW** | Named documents made into laws | *Roe v. Wade*, *GDPR* |
| **LANGUAGE** | Any named language | *English*, *Mandarin* |
| **DATE** | Absolute or relative dates/periods | *January 2020*, *next week* |
| **TIME** | Times smaller than a day | *3:00 PM*, *morning* |
| **PERCENT** | Percentages | *50%*, *ten percent* |
| **MONEY** | Monetary values | *$100*, *€50 million* |
| **QUANTITY** | Measurements | *10 kg*, *five miles* |
| **ORDINAL** | Ordinal numbers | *first*, *3rd* |
| **CARDINAL** | Numerals not covered by other types | *one*, *100*, *millions* |
| **NORP** | Nationalities, religious/political groups | *American*, *Buddhist*, *Republican* |
| **ENT** | Generic entity (used by simple regex NER) | Any capitalized phrase |
            """)
    
    logger.info(f"Launching Gradio UI on port {args.port}...")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

