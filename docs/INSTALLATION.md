# Kiln Installation Guide

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **pip** (bundled with Python)
- **Git** (for cloning the repository)
- **Node.js 18+** (for the UI; optional for backend-only usage)

### Optional for GPU-Accelerated Training (Foundry)

- **NVIDIA GPU** with CUDA 12.1+
- **CUDA Toolkit** (12.1 or later)
- **cuDNN** (matching your CUDA version)

GPU is not required for Quarry (document processing) or Forge (curriculum building). The ML classifier in Quarry Tier 1 uses scikit-learn, which runs on CPU only.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sean-murphy222/kiln.git
cd kiln

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

---

## Dependency Groups

Kiln uses optional dependency groups defined in `pyproject.toml`. Install only what you need.

### Core (always installed)

| Package | Purpose |
|---|---|
| fastapi >= 0.109.0 | Web framework |
| uvicorn[standard] >= 0.27.0 | ASGI server |
| pydantic >= 2.5.0 | Data validation |
| pdfplumber >= 0.10.0 | PDF text extraction |
| pymupdf >= 1.23.0 | PDF structural analysis (fitz) |
| python-docx >= 1.1.0 | Word document support |
| markdown-it-py >= 3.0.0 | Markdown parsing |
| tiktoken >= 0.5.0 | Token counting |
| sentence-transformers >= 2.2.0 | Embeddings |
| numpy >= 1.24.0 | Numerical operations |
| scikit-learn >= 1.4.0 | ML classifier (Tier 1) |
| joblib >= 1.3.0 | Model serialization |
| python-multipart >= 0.0.6 | File upload handling |

### Development

```bash
pip install -e ".[dev]"
```

Adds: pytest, pytest-asyncio, black, ruff, mypy

### OCR (optional, out of MVP scope)

```bash
pip install -e ".[ocr]"
```

Adds: pytesseract (requires system-level Tesseract)

### PowerPoint support (optional)

```bash
pip install -e ".[pptx]"
```

Adds: python-pptx

### Enhanced extraction with Docling (optional, Tier 2)

```bash
pip install -e ".[enhanced]"
```

Adds: docling >= 2.0.0

### AI extraction (optional, Tier 3)

```bash
pip install -e ".[ai]"
```

Adds: layoutparser, detectron2, torchvision

---

## Running the Application

### Quarry (Document Processing Server)

```bash
cd quarry
uvicorn chonk.server:app --reload --port 8420
```

The API will be available at `http://localhost:8420`. OpenAPI docs at `http://localhost:8420/docs`.

### Running Tests

```bash
# Run all tests (quarry + forge + foundry)
pytest

# Run tests for a specific module
pytest quarry/tests/
pytest forge/tests/
pytest foundry/tests/

# Run with coverage
pytest --cov=quarry/chonk --cov=forge/src --cov=foundry/src --cov-report=term-missing
```

### Linting and Formatting

```bash
# Format code
black quarry/ forge/ foundry/

# Lint
ruff check quarry/ forge/ foundry/

# Type checking
mypy quarry/chonk/ forge/src/ foundry/src/
```

---

## GPU Configuration for Foundry Training

Foundry's LoRA training pipeline can use GPU acceleration when available. The training backend is abstracted, so the pipeline works in dry-run mode on CPU for testing.

### Verifying CUDA Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `True`, GPU training is available.

### Supported Base Models

Foundry supports four base model families. Default models are pre-configured.

| Family | Default Model | Typical VRAM |
|---|---|---|
| Phi | microsoft/phi-3-mini-4k-instruct | 8 GB |
| LLaMA | meta-llama/Llama-3-8B-Instruct | 16 GB |
| Mistral | mistralai/Mistral-7B-Instruct-v0.3 | 16 GB |
| Qwen | Qwen/Qwen2-7B-Instruct | 16 GB |

For laptops with limited VRAM (8 GB or less), use quantized Phi models. For workstations with 24+ GB VRAM, LLaMA or Mistral models provide better quality.

### Troubleshooting GPU Issues

- **CUDA out of memory**: Reduce batch size in `TrainingConfig` or use a smaller base model
- **CUDA not found**: Ensure CUDA Toolkit is installed and `nvcc --version` works
- **Driver version mismatch**: Update NVIDIA drivers to match your CUDA Toolkit version
- **Docling GPU errors**: Set `DOCLING_DEVICE=cpu` to fall back to CPU mode for Tier 2 extraction

---

## Real Model Inference & Training (Opt-in)

By default Kiln runs with a **mock inference backend** and a **dry-run training
simulator**, so the full pipeline and test suite work on any machine with no GPU
and no large model downloads. To run **real** models, install the extras and
flip the backend via environment variables — no code changes required.

### Install the model extras

```bash
# Real inference (HF transformers + peft); bf16 by default
pip install -e ".[inference]"

# Real LoRA fine-tuning (peft + trl + datasets)
pip install -e ".[training]"

# Optional 4-bit quantization (bitsandbytes). May be unavailable on some
# platforms (e.g. Blackwell/sm_120 + Windows); bf16 is the default.
pip install -e ".[quant]"
```

Verify the GPU stack:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Enable real inference

| Variable | Purpose | Default |
|---|---|---|
| KILN_INFERENCE_BACKEND | `mock` or `transformers` | `mock` |
| KILN_BASE_MODEL | Base model id (required for `transformers`) | — |
| KILN_ADAPTER_PATH | LoRA adapter to attach at inference | — |
| KILN_LOAD_4BIT | `1` to load 4-bit (needs `[quant]`) | `0` (bf16) |
| KILN_INFERENCE_DTYPE | Torch dtype for the bf16/fp16 path | `bfloat16` |
| KILN_MAX_NEW_TOKENS | Generation length cap | `512` |

```bash
export KILN_INFERENCE_BACKEND=transformers
export KILN_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python kiln_server.py
```

### Enable real LoRA training

| Variable | Purpose | Default |
|---|---|---|
| FOUNDRY_TRAINING_BACKEND | `dryrun` or `real` | `dryrun` |

```bash
export FOUNDRY_TRAINING_BACKEND=real
# Run a TrainingPipeline as usual; it writes a loadable PEFT adapter to
# <output_dir>/adapter that the inference backend loads via KILN_ADAPTER_PATH.
```

The real training backend resolves LoRA target modules per model family (e.g.
Phi-3 uses `qkv_proj`/`o_proj`) and fine-tunes in bf16.

### Running the GPU/model tests

GPU-marked tests are skipped by default. Opt in with `KILN_RUN_GPU_TESTS=1`:

```bash
KILN_RUN_GPU_TESTS=1 pytest foundry/tests/test_inference_gpu.py \
                            foundry/tests/test_training_gpu.py \
                            hearth/tests/test_per_slot_routing.py
```

### VRAM guidance

Hearth loads **one real model at a time** (single-resident; loading a new slot
evicts the previous one). On 16 GB VRAM, a 0.5–3B model fits comfortably in
bf16; a 7–8B base needs 4-bit (`KILN_LOAD_4BIT=1`).

---

## Directory Structure After Installation

```
kiln/
├── quarry/
│   ├── chonk/           # Source code (import as chonk.*)
│   │   ├── tier1/       # Fingerprinting + ML classifier
│   │   ├── hierarchy/   # Document hierarchy builder
│   │   ├── qa/          # Quality assurance filters
│   │   ├── cleaning/    # Content normalization
│   │   ├── exporters/   # JSONL, CSV, JSON export + vector DB adapters
│   │   ├── enrichment/  # Metadata enrichment pipeline
│   │   ├── retrieval/   # 3-stage metadata-filtered retrieval
│   │   └── core/        # Shared data models (Block, Chunk, etc.)
│   └── tests/
├── forge/
│   ├── src/             # Curriculum builder (import as forge.src.*)
│   └── tests/
├── foundry/
│   ├── src/             # Training + evaluation (import as foundry.src.*)
│   └── tests/
├── docs/                # Documentation
└── pyproject.toml       # Project configuration
```

---

## Environment Variables

Kiln does not require any environment variables for basic operation. The following are optional.

| Variable | Purpose | Default |
|---|---|---|
| DOCLING_DEVICE | Device for Docling extraction (cpu/cuda) | auto |
| KILN_LOG_LEVEL | Logging level (DEBUG, INFO, WARNING) | INFO |

No API keys are required. Kiln is designed for fully local, offline operation after initial setup.

---

## Upgrading

```bash
git pull origin main
pip install -e ".[dev]"
```

After upgrading, re-run the test suite to verify everything works.

```bash
pytest
```
