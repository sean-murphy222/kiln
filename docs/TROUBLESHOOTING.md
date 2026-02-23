# Kiln Troubleshooting Guide

---

## Quarry Issues

### Fingerprinting fails with "not a valid PDF"

**Symptom:** `ValueError: <filename> is not a valid PDF file`

**Cause:** The file does not start with the `%PDF-` magic bytes. This can happen with:
- Scanned image files renamed to `.pdf`
- Corrupted downloads
- HTML files with `.pdf` extension

**Fix:** Verify the file is a genuine PDF. Check the first 5 bytes:
```python
with open("suspect.pdf", "rb") as f:
    print(f.read(5))  # Should be b'%PDF-'
```

### File size limit exceeded

**Symptom:** `ValueError: File size (N bytes) exceeds maximum (104857600 bytes)`

**Cause:** The PDF exceeds the default 100 MB limit.

**Fix:** Increase the limit when constructing the fingerprinter:
```python
fp = DocumentFingerprinter(max_file_size=200 * 1024 * 1024)  # 200 MB
```

### Classifier returns UNKNOWN for all documents

**Symptom:** Every `ClassificationResult.is_unknown` is `True`.

**Possible causes:**
1. The classifier has not been trained. Check `classifier.is_trained`.
2. The confidence threshold is too high. Default is 0.45; lowering it may help, but reduces precision.
3. The documents do not match any trained type profile.

**Fix:**
```python
# Check if trained
if not classifier.is_trained:
    corpus = generate_training_corpus(samples_per_type=40)
    classifier.train(corpus)

# Lower threshold if needed (trade-off: more false positives)
classifier = DocumentClassifier(confidence_threshold=0.35)
```

### Classifier accuracy below 70%

**Symptom:** Too many misclassifications.

**Fix:** This usually means the training profiles do not match your document population. Options:
1. Generate more training samples: `generate_training_corpus(samples_per_type=80)`
2. Adjust document type profiles in `taxonomy.py` to match your documents
3. Use the manual override store to correct persistent misclassifications
4. Flag for manual review per CLAUDE.md escalation procedures

### Zero fingerprint maps to REFERENCE_CARD

**Symptom:** A document with no extractable content produces a REFERENCE_CARD classification with high confidence.

**Cause:** This is expected behavior. An all-zero feature vector matches the REFERENCE_CARD profile (short documents, minimal structure). It does not indicate low confidence.

**Fix:** No fix needed. If the classification is wrong, use the manual store to override:
```python
from chonk.tier1.manual_store import ManualTypeStore
store = ManualTypeStore("overrides.json")
store.set_type("document_hash", DocumentType.FORM)
```

---

## Forge Issues

### ForgeStorageError on create

**Symptom:** `ForgeStorageError: Contributor already exists: contrib_xxx`

**Cause:** Attempting to insert a record with a duplicate primary key.

**Fix:** Use `generate_id()` for new records:
```python
contributor = Contributor(
    id=Contributor.generate_id(),  # Unique UUID-based ID
    name="Jane Smith",
)
```

### Foreign key constraint failure

**Symptom:** `ForgeStorageError: Example creation failed: ex_xxx`

**Cause:** The referenced competency_id, contributor_id, or discipline_id does not exist in the database.

**Fix:** Create parent records first:
```python
storage.create_contributor(contributor)
storage.create_discipline(discipline)
storage.create_competency(competency)
storage.create_example(example)  # Now succeeds
```

### Coverage report shows incorrect counts

**Symptom:** `get_coverage_report()` shows wrong example counts.

**Cause:** Test-set examples are excluded from coverage counts by design. The report only counts training examples (where `is_test_set = False`).

**Fix:** This is correct behavior. Test-set examples are reserved for evaluation and should not count toward curriculum coverage targets.

### Discovery session cannot advance phase

**Symptom:** `DiscoveryError` when calling `advance_phase()`.

**Cause:** Required questions for the current phase have not been answered.

**Fix:** Check progress before advancing:
```python
progress = engine.get_progress(session_id)
print(f"Phase: {progress.current_phase}")
print(f"Answered: {progress.questions_answered}/{progress.questions_total}")
# Answer remaining required questions, then advance
```

### JSONL export produces empty file

**Symptom:** The exported `.jsonl` file has 0 bytes.

**Cause:** No matching examples found. Check:
1. The discipline_id is correct
2. Examples exist for the discipline
3. If `include_test_set=False` (default), test-set examples are excluded

**Fix:**
```python
# Check what exists
training = storage.get_training_examples(discipline_id)
test = storage.get_test_set_examples(discipline_id)
print(f"Training: {len(training)}, Test: {len(test)}")
```

---

## Foundry Issues

### TrainingError: "Classifier has not been trained"

**Symptom:** `RuntimeError` when calling `predict()`.

**Fix:** Train or load a classifier before prediction:
```python
classifier = DocumentClassifier()
corpus = generate_training_corpus()
classifier.train(corpus)
# Now predict() works
```

### Training pipeline returns immediately with no result

**Symptom:** `run(dry_run=True)` completes but metrics are simulated.

**Cause:** This is expected behavior. `dry_run=True` (the default) simulates training without GPU. Real training requires a production backend (Unsloth/Axolotl).

**Fix:** For MVP testing, dry-run mode is correct. For actual training, a GPU backend must be configured (post-MVP).

### Evaluation scores all show "untested"

**Symptom:** `CompetencyRating.UNTESTED` for every competency.

**Cause:** No test cases map to the competency IDs in `competency_names`.

**Fix:** Verify test case competency_ids match the competency_names dict keys:
```python
cases = runner.load_test_cases(Path("test_set.jsonl"))
for case in cases:
    print(case.competency_id)  # Must match keys in competency_names
```

### Regression checker reports false regressions

**Symptom:** `RegressionSeverity.MAJOR` when performance actually improved.

**Cause:** The checker compares individual competency scores, not overall averages. One competency dropping while others improve triggers a regression flag.

**Fix:** Review the `competency_regressions` list in the report to see which specific competency regressed. This may be expected when curriculum changes shift focus.

### Model merge fails compatibility check

**Symptom:** `CompatibilityResult.is_compatible` is `False`.

**Cause:** Adapters have mismatched base models or incompatible LoRA configurations.

**Fix:** Verify adapters share the same base model and compatible LoRA parameters:
```python
checker = CompatibilityChecker()
result = checker.check(adapters)
print(result.issues)  # Lists specific incompatibilities
```

### Diagnostics report "convergence" issue

**Symptom:** `IssueCategory.CONVERGENCE` in the diagnostic report.

**Cause:** Training loss is not decreasing as expected.

**Fix (per plain-language guidance in the report):**
1. Check curriculum quality -- are examples clear and consistent?
2. Consider adding more examples via Forge
3. Try adjusting learning rate (lower if oscillating, higher if flat)
4. Increase number of epochs

---

## Environment Issues

### Import errors after installation

**Symptom:** `ModuleNotFoundError: No module named 'chonk'`

**Fix:** Install in editable mode from the project root:
```bash
pip install -e .
```

### Tests fail with conftest collision

**Symptom:** `import file mismatch` errors from pytest.

**Cause:** `__init__.py` files in test directories cause import collisions between quarry/tests and forge/tests.

**Fix:** Do not add `__init__.py` to test directories. This was fixed in PR #5.

### Black and ruff disagree on formatting

**Symptom:** Running black then ruff (or vice versa) produces a cycle of changes.

**Fix:** Always run ruff first, then black:
```bash
ruff check --fix quarry/ forge/ foundry/
black quarry/ forge/ foundry/
```

### Quality gate timeout

**Symptom:** The H-14 quality gate hook times out during pytest.

**Cause:** The test suite may take longer than the hook timeout (600 seconds for pytest).

**Fix:** Run tests directly to identify slow tests:
```bash
pytest --durations=10  # Show 10 slowest tests
```

---

## Performance Issues

### Fingerprinting takes over 5 seconds

**Symptom:** `DocumentFingerprinter.extract()` is slow on large PDFs.

**Cause:** The fingerprinter samples up to 50 pages. Very large documents (500+ pages) may take longer.

**Fix:** This is within expected bounds for large documents. The page sampling strategy (first 3, last 2, evenly-spaced middle) keeps analysis tractable. The 5-second target is for typical documents (< 200 pages).

### Memory usage spikes during export

**Symptom:** High memory usage when exporting large document sets.

**Cause:** All chunks are held in memory during export.

**Fix:** Export documents one at a time rather than as a project:
```python
exporter = ExporterRegistry.get_exporter("jsonl")
for doc in project.documents:
    exporter.export_document(doc, Path(f"export_{doc.id}.jsonl"))
```
