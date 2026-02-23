# Kiln API Reference

This document covers the public API for all three implemented phases: Quarry, Forge, and Foundry.

---

## Quarry -- Document Processing

### `chonk.tier1.fingerprinter.DocumentFingerprinter`

Extracts structural fingerprints from PDF documents. Analyzes raw PDF structure statistically (no content parsing beyond character-level stats) to produce a 49-feature vector for ML classification.

```python
class DocumentFingerprinter:
    def __init__(self, max_file_size: int = 104857600) -> None:
        """
        Args:
            max_file_size: Maximum allowed file size in bytes (default 100 MB).
        """

    def extract(self, path: str | Path) -> DocumentFingerprint:
        """Extract a structural fingerprint from a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            DocumentFingerprint with all sub-features populated.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid PDF or exceeds size limit.
        """
```

### `chonk.tier1.fingerprinter.DocumentFingerprint`

Container for all 6 sub-feature groups. Provides conversion to flat feature vectors.

```python
@dataclass
class DocumentFingerprint:
    byte_features: ByteLevelFeatures
    font_features: FontFeatures
    layout_features: LayoutFeatures
    character_features: CharacterFeatures
    repetition_features: RepetitionFeatures
    structural_rhythm: StructuralRhythmFeatures

    def to_feature_vector(self) -> list[float]:
        """Convert to flat list of 49 floats for ML input."""

    def feature_names(self) -> list[str]:
        """Return ordered feature names matching to_feature_vector().
        Names are prefixed with group (e.g., 'byte_file_size', 'font_size_mean').
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dictionary."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentFingerprint:
        """Deserialize from nested dictionary."""
```

**Sub-feature dataclasses (all have `to_dict()` and `from_dict()` methods):**

| Class | Features | Count |
|---|---|---|
| `ByteLevelFeatures` | file_size, pdf_version, object_count, stream_count, has_metadata, has_xmp_metadata, page_count, encrypted, has_acroform | 9 |
| `FontFeatures` | font_count, size_min, size_max, size_mean, size_std, size_median, bold_ratio, italic_ratio, monospace_ratio, distinct_sizes | 10 |
| `LayoutFeatures` | page_width, page_height, width_consistency, height_consistency, margin_left, margin_right, margin_top, margin_bottom, text_area_ratio, estimated_columns | 10 |
| `CharacterFeatures` | alpha_ratio, numeric_ratio, punctuation_ratio, whitespace_ratio, special_ratio, uppercase_ratio, total_chars | 7 |
| `RepetitionFeatures` | has_page_numbers, has_headers, has_footers, repetition_ratio, first_line_diversity | 5 |
| `StructuralRhythmFeatures` | heading_density, table_density, image_density, list_density, has_toc, toc_depth, link_count, heading_size_levels | 8 |

**Total: 49 features**

### `chonk.tier1.classifier.DocumentClassifier`

ML classifier wrapping scikit-learn's GradientBoostingClassifier.

```python
class DocumentClassifier:
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.45

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        n_estimators: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None: ...

    @property
    def is_trained(self) -> bool:
        """True if the classifier has been trained."""

    def train(self, corpus: TrainingCorpus) -> TrainingReport:
        """Fit the model on a training corpus.

        Raises:
            ValueError: If corpus is empty or has wrong feature count.
        """

    def predict(self, fingerprint: DocumentFingerprint) -> ClassificationResult:
        """Classify a single document fingerprint.

        Raises:
            RuntimeError: If classifier has not been trained.
            ValueError: If feature vector has wrong length.
        """

    def predict_batch(
        self, fingerprints: list[DocumentFingerprint]
    ) -> list[ClassificationResult]:
        """Classify multiple fingerprints."""

    def feature_importances(self) -> list[FeatureImportance]:
        """Return features ranked by importance, highest first."""

    def save(self, path: str | Path) -> None:
        """Persist trained classifier using joblib.
        WARNING: Do not load from untrusted sources.
        """

    @classmethod
    def load(cls, path: str | Path) -> DocumentClassifier:
        """Load a trained classifier from disk.
        WARNING: Do not load from untrusted sources.
        """
```

### `chonk.tier1.classifier.ClassificationResult`

```python
@dataclass
class ClassificationResult:
    document_type: DocumentType    # Predicted type (UNKNOWN if low confidence)
    confidence: float              # Probability of predicted class (0.0-1.0)
    probabilities: dict[str, float]  # All class probabilities
    is_unknown: bool               # True if confidence below threshold

    def to_dict(self) -> dict[str, Any]: ...
```

### `chonk.tier1.taxonomy.DocumentType`

```python
class DocumentType(str, Enum):
    TECHNICAL_MANUAL = "technical_manual"
    MAINTENANCE_PROCEDURE = "maintenance_procedure"
    PARTS_CATALOG = "parts_catalog"
    WIRING_DIAGRAM = "wiring_diagram"
    REGULATION = "regulation"
    SPECIFICATION = "specification"
    TRAINING_MATERIAL = "training_material"
    FORM = "form"
    REPORT = "report"
    REFERENCE_CARD = "reference_card"
    ACADEMIC_PAPER = "academic_paper"
    PRESENTATION = "presentation"
    DATASHEET = "datasheet"
    CONTRACT = "contract"
    UNKNOWN = "unknown"          # Sentinel, not trainable
```

`TRAINABLE_TYPES` is a list of all types except UNKNOWN.

### `chonk.exporters.schema.ChonkRecord`

Canonical export record for a single chunk. See [EXPORT_FORMAT.md](EXPORT_FORMAT.md) for full schema details.

```python
@dataclass
class ChonkRecord:
    id: str
    content: str
    token_count: int
    hierarchy_path: str = ""
    quality_score: float = 1.0
    source: str = ""
    source_type: str = ""
    document_id: str = ""
    page_start: int | None = None
    page_end: int | None = None
    enrichment_fields: dict[str, Any] = field(default_factory=dict)
    user_metadata: dict[str, Any] = field(default_factory=dict)
    system_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize with schema_version included."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChonkRecord: ...
```

### `chonk.exporters.schema.VectorDBAdapter`

Static methods to transform ChonkRecords into vector-DB-specific payloads.

```python
class VectorDBAdapter:
    @staticmethod
    def to_chromadb(record: ChonkRecord) -> dict[str, Any]:
        """Returns dict with keys: id, document, metadata."""

    @staticmethod
    def to_qdrant(record: ChonkRecord) -> dict[str, Any]:
        """Returns dict with keys: id, payload."""

    @staticmethod
    def to_weaviate(record: ChonkRecord) -> dict[str, Any]:
        """Returns dict with keys: class, properties."""

    @staticmethod
    def to_pinecone(record: ChonkRecord) -> dict[str, Any]:
        """Returns dict with keys: id, metadata."""
```

### `chonk.exporters.base.ExporterRegistry`

```python
class ExporterRegistry:
    @classmethod
    def get_exporter(cls, name: str) -> BaseExporter | None: ...

    @classmethod
    def available_exporters(cls) -> list[str]: ...

    @classmethod
    def export_document(cls, document: ChonkDocument, path: Path, format: str) -> Path: ...

    @classmethod
    def export_project(cls, project: ChonkProject, path: Path, format: str) -> Path: ...
```

---

## Forge -- Curriculum Builder

### `forge.src.models` -- Data Models

#### Contributor

```python
@dataclass
class Contributor:
    id: str                         # Prefixed with 'contrib_'
    name: str
    email: str = ""
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def generate_id() -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contributor: ...
```

#### Discipline

```python
@dataclass
class Discipline:
    id: str                         # Prefixed with 'disc_'
    name: str
    description: str
    status: DisciplineStatus        # DRAFT, ACTIVE, ARCHIVED
    created_by: str                 # Contributor ID
    vocabulary: list[str]
    document_types: list[str]
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def generate_id() -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Discipline: ...
```

#### Competency

```python
@dataclass
class Competency:
    id: str                         # Prefixed with 'comp_'
    name: str
    description: str
    discipline_id: str
    parent_id: str | None = None    # For hierarchical competencies
    coverage_target: int = 25       # Examples needed for full coverage
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def generate_id() -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Competency: ...
```

#### Example

```python
@dataclass
class Example:
    id: str                         # Prefixed with 'ex_'
    question: str                   # Instruction field in Alpaca format
    ideal_answer: str               # Output field in Alpaca format
    competency_id: str
    contributor_id: str
    discipline_id: str
    variants: list[str]             # Alternative question phrasings
    context: str = ""
    review_status: ReviewStatus     # PENDING, APPROVED, REJECTED, NEEDS_REVISION
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    is_test_set: bool = False       # Held out for evaluation
    created_at: datetime
    updated_at: datetime

    def to_training_record(self) -> dict[str, Any]:
        """Convert to Alpaca format: {instruction, input, output, metadata}."""
```

#### Enums

```python
class ContributorRole(str, Enum):   # CONTRIBUTOR, LEAD, ADMIN
class DisciplineStatus(str, Enum):  # DRAFT, ACTIVE, ARCHIVED
class ReviewStatus(str, Enum):      # PENDING, APPROVED, REJECTED, NEEDS_REVISION
class CurriculumStatus(str, Enum):  # DRAFT, PUBLISHED
class ResponseType(str, Enum):      # FREE_TEXT, LIST_ITEMS, SCALE_1_5
class DiscoveryPhase(str, Enum):    # ORIENTATION, DOCUMENTS, COMPETENCIES, VOCABULARY
class SessionStatus(str, Enum):     # IN_PROGRESS, COMPLETED, ABANDONED
```

### `forge.src.storage.ForgeStorage`

SQLite-backed storage for all Forge domain models.

```python
class ForgeStorage:
    def __init__(self, db_path: str | Path = ":memory:") -> None: ...
    def __enter__(self) -> ForgeStorage: ...
    def __exit__(self, *args) -> None: ...
    def close(self) -> None: ...
    def initialize_schema(self) -> None: ...

    # Contributors
    def create_contributor(self, contributor: Contributor) -> Contributor: ...
    def get_contributor(self, contributor_id: str) -> Contributor | None: ...
    def update_contributor(self, contributor: Contributor) -> Contributor: ...
    def delete_contributor(self, contributor_id: str) -> bool: ...

    # Disciplines
    def create_discipline(self, discipline: Discipline) -> Discipline: ...
    def get_discipline(self, discipline_id: str) -> Discipline | None: ...
    def get_all_disciplines(self, status: DisciplineStatus | None = None) -> list[Discipline]: ...
    def update_discipline(self, discipline: Discipline) -> Discipline: ...
    def delete_discipline(self, discipline_id: str) -> bool: ...

    # Competencies
    def create_competency(self, competency: Competency) -> Competency: ...
    def get_competency(self, competency_id: str) -> Competency | None: ...
    def get_competencies_for_discipline(self, discipline_id: str) -> list[Competency]: ...
    def update_competency(self, competency: Competency) -> Competency: ...
    def delete_competency(self, competency_id: str) -> bool: ...

    # Examples
    def create_example(self, example: Example) -> Example: ...
    def get_example(self, example_id: str) -> Example | None: ...
    def get_examples_for_competency(
        self, competency_id: str, include_test_set: bool = True
    ) -> list[Example]: ...
    def get_training_examples(self, discipline_id: str) -> list[Example]: ...
    def get_test_set_examples(self, discipline_id: str) -> list[Example]: ...
    def update_example(self, example: Example) -> Example: ...
    def delete_example(self, example_id: str) -> bool: ...

    # Discipline-Contributor Associations
    def add_contributor_to_discipline(self, dc: DisciplineContributor) -> DisciplineContributor: ...
    def get_discipline_contributors(self, discipline_id: str) -> list[DisciplineContributor]: ...
    def update_contributor_in_discipline(self, dc: DisciplineContributor) -> DisciplineContributor: ...
    def remove_contributor_from_discipline(self, discipline_id: str, contributor_id: str) -> bool: ...

    # Curriculum Versioning
    def create_curriculum_version(self, discipline_id: str, created_by: str) -> CurriculumVersion: ...
    def get_latest_curriculum_version(self, discipline_id: str) -> CurriculumVersion | None: ...
    def publish_curriculum_version(self, version_id: str) -> CurriculumVersion: ...
    def get_curriculum_history(self, discipline_id: str) -> list[CurriculumVersion]: ...

    # Coverage Report
    def get_coverage_report(self, discipline_id: str) -> dict[str, Any]:
        """Returns dict with: discipline_id, total_examples, total_test_examples,
        competency_coverage, gaps, coverage_complete."""

    # JSONL Export
    def export_to_jsonl(
        self, discipline_id: str, output_path: str | Path, include_test_set: bool = False
    ) -> Path: ...
    def export_test_set_jsonl(self, discipline_id: str, output_path: str | Path) -> Path: ...

    # Discovery Sessions
    def save_discovery_session(self, session: DiscoverySession) -> DiscoverySession: ...
    def get_discovery_session(self, session_id: str) -> DiscoverySession | None: ...
    def list_discovery_sessions(
        self, contributor_id: str | None = None, status: SessionStatus | None = None
    ) -> list[DiscoverySession]: ...
```

### `forge.src.discovery.DiscoveryEngine`

Manages the discipline discovery interview workflow.

```python
class DiscoveryEngine:
    def __init__(self, storage: ForgeStorage) -> None: ...

    def start_session(self, discipline_name: str, contributor_id: str) -> DiscoverySession: ...
    def record_response(
        self, session_id: str, question_id: str, raw_text: str, items: list[str] = ...,
        scale_value: int | None = None
    ) -> DiscoverySession: ...
    def advance_phase(self, session_id: str) -> DiscoverySession: ...
    def complete_session(self, session_id: str) -> DiscoverySession: ...
    def get_progress(self, session_id: str) -> SessionProgress: ...
    def abandon_session(self, session_id: str) -> DiscoverySession: ...
```

### `forge.src.competency.CompetencyMapper`

Engine for refining and organizing competency maps.

```python
class CompetencyMapper:
    def __init__(self, storage: ForgeStorage) -> None: ...

    def load_from_discovery(
        self, discipline_id: str, session: DiscoverySession
    ) -> list[Competency]: ...
    def add_competency(
        self, discipline_id: str, name: str, description: str,
        parent_id: str | None = None, coverage_target: int = 25
    ) -> Competency: ...
    def update_competency(
        self, competency_id: str, name: str | None = None,
        description: str | None = None, coverage_target: int | None = None
    ) -> Competency: ...
    def set_parent(self, competency_id: str, parent_id: str | None) -> Competency: ...
    def remove_competency(self, competency_id: str) -> bool: ...
    def get_coverage_summary(self, discipline_id: str) -> CompetencyMapSummary: ...
    def get_competency_tree(self, discipline_id: str) -> list[dict[str, Any]]: ...
    def finalize_map(self, discipline_id: str) -> CompetencyMapSummary: ...
```

### `forge.src.examples.ExampleElicitor`

Guides experts through creating training examples.

```python
class ExampleElicitor:
    def __init__(self, storage: ForgeStorage) -> None: ...

    def start_session(
        self, discipline_id: str, contributor_id: str, competency_ids: list[str] = ...
    ) -> ElicitationSession: ...
    def create_draft(
        self, session_id: str, question: str, ideal_answer: str,
        competency_id: str, reasoning_pattern: ReasoningPattern = ...,
        context: str = "", variants: list[str] = ...
    ) -> ExampleDraft: ...
    def update_draft(
        self, session_id: str, draft_id: str, question: str | None = None,
        ideal_answer: str | None = None, ...
    ) -> ExampleDraft: ...
    def finalize_draft(self, session_id: str, draft_id: str) -> Example: ...
    def suggest_competencies(self, discipline_id: str) -> list[CompetencySuggestion]: ...
    def pause_session(self, session_id: str) -> ElicitationSession: ...
    def resume_session(self, session_id: str) -> ElicitationSession: ...
    def complete_session(self, session_id: str) -> ElicitationSession: ...
```

---

## Foundry -- Training and Evaluation

### `foundry.src.training.TrainingPipeline`

Complete LoRA training workflow orchestrator.

```python
class TrainingPipeline:
    def __init__(
        self, config: TrainingConfig, registry: TrainingRegistry | None = None
    ) -> None: ...

    def prepare(self, curriculum_path: str | Path) -> dict[str, Any]:
        """Load, validate, and split curriculum data. Returns preparation summary."""

    def run(self, dry_run: bool = True) -> TrainingResult:
        """Execute training. Use dry_run=True for testing without GPU."""

    def cancel(self) -> None:
        """Cancel a running training job."""

    def save_result(self, output_dir: str | Path) -> Path:
        """Save training result and adapter to output directory."""
```

### `foundry.src.training.TrainingConfig`

```python
@dataclass
class TrainingConfig:
    base_model_family: BaseModelFamily = BaseModelFamily.PHI
    base_model_name: str | None = None   # Uses default for family if None
    lora_config: LoRAConfig = ...
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    validation_split: float = 0.1
    max_sequence_length: int = 2048
    gradient_accumulation_steps: int = 4
    seed: int = 42
```

### `foundry.src.training.LoRAConfig`

```python
@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = ...   # Default: ["q_proj", "v_proj"]
    bias: str = "none"
```

### `foundry.src.training.BaseModelFamily`

```python
class BaseModelFamily(str, Enum):
    PHI = "phi"           # microsoft/phi-3-mini-4k-instruct
    LLAMA = "llama"       # meta-llama/Llama-3-8B-Instruct
    MISTRAL = "mistral"   # mistralai/Mistral-7B-Instruct-v0.3
    QWEN = "qwen"         # Qwen/Qwen2-7B-Instruct
```

### `foundry.src.training.CurriculumLoader`

```python
class CurriculumLoader:
    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """Load Alpaca-format JSONL file."""

    def validate_record(self, record: dict[str, Any]) -> bool:
        """Check record has required Alpaca fields."""

    def split_train_val(
        self, records: list[dict], val_ratio: float = 0.1, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        """Split records into train/validation sets."""

    def get_statistics(self, records: list[dict]) -> dict[str, Any]:
        """Compute curriculum statistics."""
```

### `foundry.src.evaluation.EvaluationRunner`

```python
class EvaluationRunner:
    def __init__(self, config: EvaluationConfig | None = None) -> None: ...

    def run_evaluation(
        self, model: ModelInference, test_cases: list[TestCase],
        competency_names: dict[str, str], model_name: str = "",
        discipline_id: str = ""
    ) -> EvaluationReport: ...

    def run_comparison(
        self, model_a: ModelInference, model_b: ModelInference,
        test_cases: list[TestCase], competency_names: dict[str, str],
        name_a: str = "Model A", name_b: str = "Model B",
        discipline_id: str = ""
    ) -> ModelComparison: ...

    def load_test_cases(self, path: Path) -> list[TestCase]: ...
```

### `foundry.src.rag_integration.RAGPipeline`

```python
class RAGPipeline:
    def __init__(
        self, model: ModelInference, retrieval: RetrievalAdapter,
        config: RAGConfig | None = None
    ) -> None: ...

    def query(self, query: str) -> RAGResponse:
        """Execute RAG pipeline: retrieve -> build context -> generate -> cite."""

    def batch_query(self, queries: list[str]) -> list[RAGResponse]:
        """Process multiple queries sequentially."""
```

### `foundry.src.regression.RegressionChecker`

```python
class RegressionChecker:
    def __init__(self, config: RegressionConfig | None = None) -> None: ...

    def compare(
        self, baseline: EvaluationReport, current: EvaluationReport,
        change_type: ChangeType
    ) -> RegressionReport:
        """Compare two evaluation reports and detect regressions."""
```

### `foundry.src.regression.VersionManager`

```python
class VersionManager:
    def __init__(self, registry_path: str | Path) -> None: ...

    def register_version(
        self, discipline_id: str, model_name: str, adapter_path: str,
        evaluation_report_id: str = "", change_type: ChangeType = ...,
        notes: str = ""
    ) -> VersionEntry: ...
    def get_version(self, version_id: str) -> VersionEntry | None: ...
    def list_versions(self, discipline_id: str) -> list[VersionEntry]: ...
    def get_active_version(self, discipline_id: str) -> VersionEntry | None: ...
    def set_active(self, version_id: str) -> VersionEntry: ...
    def rollback(self, discipline_id: str) -> VersionEntry | None: ...
    def get_version_history(self, discipline_id: str) -> list[VersionEntry]: ...
```

### `foundry.src.merging.MergePipeline`

```python
class MergePipeline:
    def __init__(self, config: MergeConfig | None = None) -> None: ...

    def merge(self, adapters: list[AdapterInfo]) -> MergeResult:
        """Merge multiple LoRA adapters into one."""

    def get_status(self) -> MergeStatus: ...
```

### `foundry.src.merging.MergeConfig`

```python
@dataclass
class MergeConfig:
    method: MergeMethod = MergeMethod.LINEAR  # LINEAR or TIES
    weights: list[float] | None = None        # Per-adapter weights (defaults to equal)
    density: float = 0.5                      # TIES sparsity parameter
    output_name: str = ""
```

### `foundry.src.diagnostics.TrainingDiagnostics`

```python
class TrainingDiagnostics:
    def __init__(self, config: DiagnosticConfig | None = None) -> None: ...

    def analyze_training(self, metrics: list[MetricSnapshot]) -> DiagnosticReport:
        """Analyze training metrics for convergence, overfitting, instability."""

    def analyze_curriculum(self, curriculum_stats: dict[str, Any]) -> DiagnosticReport:
        """Analyze curriculum quality (size, balance, diversity)."""

    def analyze_full(
        self, metrics: list[MetricSnapshot], curriculum_stats: dict[str, Any]
    ) -> DiagnosticReport:
        """Run both training and curriculum analysis."""
```

### `foundry.src.diagnostics.MetricSnapshot`

```python
@dataclass
class MetricSnapshot:
    epoch: int
    step: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float | None = None
    timestamp: datetime | None = None
```
