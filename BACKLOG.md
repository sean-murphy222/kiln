# Kiln Sprint Backlog

## Sprint Goal
**MVP Timeline:** 22 weeks (11 two-week sprints) | Part-time 15-20 hours/week
**Target:** End-to-end demonstrable platform: Quarry → Forge → Foundry → Hearth

---

## PHASE 1: QUARRY COMPLETION (Sprints 1-3, 6 weeks)

### Sprint 1: Structural Fingerprinting Foundation

- [ ] **T-001** | P1 | 8 pts | Sprint 1
  **Title:** Implement Tier 1 statistical document analysis
  **Description:** Build the statistical fingerprinting system that analyzes raw document structure without parsing content. Extract byte patterns, formatting markers, whitespace distributions, character frequency profiles, repetition patterns, and structural element rhythm.
  **Acceptance Criteria:**
  - [ ] Feature extraction pipeline functional
  - [ ] Produces consistent structural fingerprints for same document type
  - [ ] Processing time < 5 seconds per document
  **Files:** quarry/src/tier1/fingerprinter.py, quarry/tests/test_fingerprinter.py
  **Depends On:** None
  **Blocked By:** None

- [ ] **T-002** | P1 | 5 pts | Sprint 1
  **Title:** Bootstrap ML classifier with open-source dataset
  **Description:** Initialize document type classifier using Docling's training corpus and open-source document layout datasets. Implement random forest or gradient boost classifier (NOT LLM).
  **Acceptance Criteria:**
  - [ ] Classifier trained on initial corpus (500+ documents, 20-30 types)
  - [ ] Baseline accuracy >70% on known document types
  - [ ] Feature importance inspectable
  - [ ] Inference time < 1 second per document
  **Files:** quarry/src/tier1/classifier.py, quarry/src/tier1/bootstrap_data.py
  **Depends On:** T-001
  **Blocked By:** None

- [ ] **T-003** | P2 | 3 pts | Sprint 1
  **Title:** Manual classification fallback workflow
  **Description:** Implement fallback mode for novel document types classifier hasn't seen. User identifies document type/conventions; system logs as new training example.
  **Acceptance Criteria:**
  - [ ] UI for manual document type input
  - [ ] Structural conventions form functional
  - [ ] New examples added to classifier training queue
  - [ ] Graceful degradation from unknown type
  **Files:** quarry/src/tier1/manual_classifier.py, ui/src/components/ManualClassification.tsx
  **Depends On:** T-002
  **Blocked By:** None

### Sprint 2: Hierarchy Construction & QA

- [ ] **T-004** | P1 | 8 pts | Sprint 2
  **Title:** Complete Tier 3 hierarchy construction
  **Description:** Finish the remaining 30% of hierarchy building. Ensure all heading levels properly nested, section numbering tracked, parent-child relationships correct.
  **Acceptance Criteria:**
  - [ ] Handles 6 heading levels
  - [ ] Section numbering schemes detected (1.1, 1.1.1, etc.)
  - [ ] Orphaned headings prevented
  - [ ] Works with Tier 1 structural profile as input
  **Files:** quarry/src/tier3/hierarchy.py, quarry/tests/test_hierarchy.py
  **Depends On:** T-002 (uses structural profile)
  **Blocked By:** None

- [ ] **T-005** | P1 | 5 pts | Sprint 2
  **Title:** Implement classification and filtering QA pass
  **Description:** Build system to identify and remove zero-value content: TOCs, indices, distribution statements, boilerplate, page headers.
  **Acceptance Criteria:**
  - [ ] Pattern library for common boilerplate
  - [ ] Repetition detection algorithm
  - [ ] Configurable filtering rules per document type
  - [ ] Filtered content logged for review
  **Files:** quarry/src/tier3/filtering.py, quarry/tests/test_filtering.py
  **Depends On:** T-004
  **Blocked By:** None

- [ ] **T-006** | P1 | 5 pts | Sprint 2
  **Title:** Implement cleaning and normalization
  **Description:** Clean chunks with value but structural noise: strip repetitive headers, normalize whitespace, consolidate continuations, remove formatting artifacts.
  **Acceptance Criteria:**
  - [ ] Whitespace normalization consistent
  - [ ] Header repetition removal functional
  - [ ] Continuation entries consolidated
  - [ ] LaTeX/formatting artifacts cleaned
  **Files:** quarry/src/tier3/cleaning.py, quarry/tests/test_cleaning.py
  **Depends On:** T-005
  **Blocked By:** None

### Sprint 3: Metadata Enrichment & Retrieval Integration

- [ ] **T-007** | P1 | 8 pts | Sprint 3
  **Title:** Build metadata enrichment pipeline
  **Description:** Extract structured metadata from formatting cues: headings, subheadings, section markers. Derive domain-specific metadata where formatting permits (equipment system, maintenance level, case citations, etc.).
  **Acceptance Criteria:**
  - [ ] Metadata fields defined per document type
  - [ ] Extraction rules configurable
  - [ ] Metadata validation and quality scoring
  - [ ] Export format separates body content from metadata
  **Files:** quarry/src/tier3/metadata.py, quarry/tests/test_metadata.py
  **Depends On:** T-006
  **Blocked By:** None

- [ ] **T-008** | P1 | 8 pts | Sprint 3
  **Title:** Implement metadata-filtered retrieval pipeline
  **Description:** Build 3-stage retrieval: (1) structural pre-filter on metadata, (2) semantic search within filtered set, (3) validation pass. Integrate with vector database.
  **Acceptance Criteria:**
  - [ ] Stage 1 deterministic filter functional (regex, rule-based)
  - [ ] Stage 2 embedding search on filtered subset
  - [ ] Stage 3 validation against structural patterns
  - [ ] Performance benchmarks show pre-filter reduces search space 80%+
  **Files:** quarry/src/retrieval/pipeline.py, quarry/tests/test_retrieval.py
  **Depends On:** T-007
  **Blocked By:** None

- [ ] **T-009** | P2 | 3 pts | Sprint 3
  **Title:** Export format standardization and documentation
  **Description:** Document Quarry output JSON schema. Provide mapping examples for ChromaDB, Qdrant, Weaviate, Pinecone. Ensure vector-database-agnostic.
  **Acceptance Criteria:**
  - [ ] JSON schema documented
  - [ ] Mapping examples for 4 major vector DBs
  - [ ] Export format versioning implemented
  - [ ] Sample exports included in repo
  **Files:** quarry/docs/export-format.md, quarry/examples/
  **Depends On:** T-008
  **Blocked By:** None

**MILESTONE:** ✓ Quarry MVP Complete
- PDF in → clean, classified, metadata-enriched hierarchical chunks out
- Retrieval-ready knowledge bases
- Portable export for external pipelines

---

## PHASE 2: FORGE CORE FRAMEWORK (Sprints 4-7, 8 weeks)

### Sprint 4: Data Architecture & Discipline Discovery

- [ ] **T-010** | P1 | 8 pts | Sprint 4
  **Title:** Design and implement Forge data model
  **Description:** Define schemas for disciplines, competencies, examples, curricula. Implement storage layer (SQLite for MVP). Design for multi-contributor support.
  **Acceptance Criteria:**
  - [ ] Database schema covers all Forge entities
  - [ ] CRUD operations functional
  - [ ] Multi-contributor support designed in
  - [ ] Export to JSONL for Foundry
  **Files:** forge/src/models.py, forge/src/storage.py, forge/tests/test_storage.py
  **Depends On:** None
  **Blocked By:** None

- [ ] **T-011** | P1 | 8 pts | Sprint 4
  **Title:** Build discipline discovery interview framework
  **Description:** Create structured questionnaire system for Step 1 (Discipline Discovery). Framework-only, no LLM. Templates and forms guide expert through surfacing discipline characteristics.
  **Acceptance Criteria:**
  - [ ] Question templates cover all discipline aspects
  - [ ] Response capture and structuring functional
  - [ ] Discipline model generated from responses
  - [ ] Session resumable (save/load state)
  - [ ] Estimated completion time 45-60 minutes per discipline
  **Files:** forge/src/discovery.py, forge/tests/test_discovery.py, ui/src/components/Discovery/
  **Depends On:** T-010
  **Blocked By:** None

- [ ] **T-012** | P2 | 3 pts | Sprint 4
  **Title:** Create discipline model visualization
  **Description:** Build UI to display structured discipline model after discovery session. Shows document types, competencies, question categories, vocabulary, patterns.
  **Acceptance Criteria:**
  - [ ] Visual representation of discipline model
  - [ ] Editable (expert can refine)
  - [ ] Export to JSON
  **Files:** ui/src/components/DisciplineModel.tsx
  **Depends On:** T-011
  **Blocked By:** None

### Sprint 5: Competency Mapping & Coverage Analysis

- [ ] **T-013** | P1 | 8 pts | Sprint 5
  **Title:** Build competency mapping system
  **Description:** Translate discipline model into competency map (Step 2). Framework for defining competency areas, tagging examples, tracking coverage.
  **Acceptance Criteria:**
  - [ ] Competency categories generated from discipline model
  - [ ] Expert can validate, refine, add competencies
  - [ ] Hierarchy support (parent/child competencies)
  - [ ] Coverage tracking per competency
  - [ ] Estimated completion time 15-20 minutes
  **Files:** forge/src/competency.py, forge/tests/test_competency.py
  **Depends On:** T-011
  **Blocked By:** None

- [ ] **T-014** | P1 | 5 pts | Sprint 5
  **Title:** Implement real-time coverage analysis
  **Description:** Track which competency areas have sufficient examples, which need more. Visual dashboard shows coverage gaps.
  **Acceptance Criteria:**
  - [ ] Coverage metrics per competency
  - [ ] Visual dashboard (heatmap or progress bars)
  - [ ] Recommendations for which areas need attention
  - [ ] Updates in real-time as examples added
  **Files:** forge/src/coverage.py, ui/src/components/CoverageDashboard.tsx
  **Depends On:** T-013
  **Blocked By:** None

- [ ] **T-015** | P2 | 5 pts | Sprint 5
  **Title:** Build example elicitation interface (Step 3 foundation)
  **Description:** Create UI for Step 3 (Example Elicitation). Expert enters question, ideal answer, tricky variants, explanations. Tag with competency, reasoning pattern.
  **Acceptance Criteria:**
  - [ ] Form for question/answer/variants entry
  - [ ] Competency area selector
  - [ ] Reasoning pattern tagging
  - [ ] Metadata fields (equipment, procedure, etc.)
  - [ ] Save draft, resume later
  **Files:** ui/src/components/ExampleElicitation.tsx, forge/src/examples.py
  **Depends On:** T-013
  **Blocked By:** None

### Sprint 6: Quality Scaffolding & Multi-Contributor

- [ ] **T-016** | P1 | 8 pts | Sprint 6
  **Title:** Implement consistency checking (Step 4)
  **Description:** Enforce consistency across growing curriculum. Check new examples against established patterns. Flag cross-contributor inconsistencies.
  **Acceptance Criteria:**
  - [ ] Response length consistency checking
  - [ ] Terminology consistency across examples
  - [ ] Citation format consistency
  - [ ] Flag conflicting examples for review
  - [ ] Suggest edits to maintain consistency
  **Files:** forge/src/consistency.py, forge/tests/test_consistency.py
  **Depends On:** T-015
  **Blocked By:** None

- [ ] **T-017** | P1 | 5 pts | Sprint 6
  **Title:** Build multi-contributor workflow
  **Description:** Support multiple SMEs contributing to same discipline. Discipline lead role for reviewing contributions and resolving conflicts.
  **Acceptance Criteria:**
  - [ ] User roles: contributor, lead, admin
  - [ ] Ownership per competency area
  - [ ] Review queue for discipline lead
  - [ ] Conflict resolution workflow
  - [ ] Contribution attribution tracked
  **Files:** forge/src/contributors.py, forge/src/auth.py
  **Depends On:** T-016
  **Blocked By:** None

- [ ] **T-018** | P1 | 5 pts | Sprint 6
  **Title:** Implement held-out test set reservation
  **Description:** Automatically reserve percentage of examples per competency as held-out test set for Foundry evaluation. Expert also provides challenge examples.
  **Acceptance Criteria:**
  - [ ] Configurable percentage (default 15-20%)
  - [ ] Stratified sampling per competency
  - [ ] Challenge examples explicitly marked
  - [ ] Test set never shown during training
  - [ ] Export separate training/test JSONL files
  **Files:** forge/src/test_split.py, forge/tests/test_split.py
  **Depends On:** T-015
  **Blocked By:** None

### Sprint 7: Quarry Integration & First Curriculum

- [ ] **T-019** | P1 | 8 pts | Sprint 7
  **Title:** Integrate Quarry for example scaffolding
  **Description:** Allow Forge to leverage Quarry-processed documents to scaffold candidate examples. Expert reviews, edits, validates.
  **Acceptance Criteria:**
  - [ ] Quarry knowledge base browser in Forge
  - [ ] Generate candidate Q/A from chunk + metadata
  - [ ] Expert reviews and edits before accepting
  - [ ] Quarry source tracked for provenance
  **Files:** forge/src/quarry_integration.py, forge/tests/test_quarry_integration.py
  **Depends On:** T-008, T-015
  **Blocked By:** None

- [ ] **T-020** | P1 | 13 pts | Sprint 7
  **Title:** Create first validated discipline curriculum
  **Description:** Run full Forge workflow with real domain expert (military maintenance). Produce 300-500 example curriculum covering full competency range. This is independently demonstrable MVP deliverable.
  **Acceptance Criteria:**
  - [ ] Discipline discovery session completed
  - [ ] Competency map validated
  - [ ] 300-500 examples across all competencies
  - [ ] Quality scaffolding passed
  - [ ] Exported as training/test JSONL
  - [ ] Session data captured for facilitator model training
  **Files:** forge/data/military-maintenance-curriculum.jsonl
  **Depends On:** T-011, T-013, T-015, T-016, T-018
  **Blocked By:** None

**MILESTONE:** ✓ Forge Framework Operational
- Domain expert can create validated curriculum without ML expertise
- First real curriculum exists for military maintenance
- Multi-contributor support functional
- Quarry integration working

---

## PHASE 3: FOUNDRY + INTEGRATION (Sprints 8-10, 6 weeks)

### Sprint 8: Training Pipeline & Base Evaluation

- [ ] **T-021** | P1 | 8 pts | Sprint 8
  **Title:** Implement LoRA training pipeline
  **Description:** Build training workflow using Unsloth or Axolotl. Take Forge JSONL + base model → produce trained LoRA. Sensible defaults, advanced settings optional.
  **Acceptance Criteria:**
  - [ ] Training pipeline functional end-to-end
  - [ ] Base model selection guidance (Phi, Llama, Mistral, Qwen)
  - [ ] Hyperparameters auto-configured from curriculum size
  - [ ] Progress monitoring and logging
  - [ ] Trained LoRA export (10-100MB)
  **Files:** foundry/src/training.py, foundry/tests/test_training.py
  **Depends On:** T-020 (needs curriculum)
  **Blocked By:** None

- [ ] **T-022** | P1 | 8 pts | Sprint 8
  **Title:** Build competency-based evaluation system
  **Description:** Implement Layer 1 (competency testing) and Layer 2 (comparative evaluation). Run held-out test set, report per-competency accuracy. Compare LoRA vs base model.
  **Acceptance Criteria:**
  - [ ] Competency test execution on held-out set
  - [ ] Results reported per competency (plain language)
  - [ ] Side-by-side base vs LoRA comparison
  - [ ] Visual dashboard showing results
  - [ ] Detailed logs per test case
  **Files:** foundry/src/evaluation.py, foundry/tests/test_evaluation.py, ui/src/components/EvaluationDashboard.tsx
  **Depends On:** T-021
  **Blocked By:** None

- [ ] **T-023** | P2 | 5 pts | Sprint 8
  **Title:** Implement failure detection and guidance
  **Description:** Auto-detect training issues (loss not converging, overfitting). Provide plain-language guidance for fixes.
  **Acceptance Criteria:**
  - [ ] Loss curve monitoring
  - [ ] Overfitting detection (train vs validation accuracy)
  - [ ] Actionable guidance (not ML jargon)
  - [ ] Suggested fixes linked to Forge (add examples to weak areas)
  **Files:** foundry/src/diagnostics.py, foundry/tests/test_diagnostics.py
  **Depends On:** T-021
  **Blocked By:** None

### Sprint 9: RAG Integration & Regression Testing

- [ ] **T-024** | P1 | 13 pts | Sprint 9
  **Title:** Integrate LoRA with Quarry retrieval pipeline
  **Description:** Connect Foundry-trained LoRA to Quarry knowledge base. Implement RAG-integrated evaluation (Layer 3). End-to-end query → retrieval → generation → citation.
  **Acceptance Criteria:**
  - [ ] LoRA loads and runs inference
  - [ ] Quarry retrieval pipeline integrated
  - [ ] Citations back to source documents functional
  - [ ] End-to-end query testing working
  - [ ] Accuracy measured on realistic questions
  **Files:** foundry/src/rag_integration.py, foundry/tests/test_rag_integration.py
  **Depends On:** T-008, T-022
  **Blocked By:** None

- [ ] **T-025** | P1 | 5 pts | Sprint 9
  **Title:** Build regression testing system
  **Description:** Store evaluation runs with timestamps. Auto-trigger on changes (retrain, merge, base model swap, Quarry reprocess). Flag competency regressions.
  **Acceptance Criteria:**
  - [ ] Evaluation history stored
  - [ ] Version comparison UI (green/yellow/red)
  - [ ] Auto-trigger on all relevant events
  - [ ] Rollback to previous version functional
  - [ ] Regression alerts actionable
  **Files:** foundry/src/regression.py, foundry/tests/test_regression.py
  **Depends On:** T-022
  **Blocked By:** None

- [ ] **T-026** | P2 | 3 pts | Sprint 9
  **Title:** Implement model merging (optional)
  **Description:** Support linear or TIES merging of multiple discipline LoRAs. Fast (minutes), no retraining. Auto-evaluate merged model.
  **Acceptance Criteria:**
  - [ ] Linear merging functional
  - [ ] TIES merging functional
  - [ ] Merged model evaluated against both source test suites
  - [ ] Accuracy tradeoff clearly presented
  - [ ] User can choose merged or individual models
  **Files:** foundry/src/merging.py, foundry/tests/test_merging.py
  **Depends On:** T-022
  **Blocked By:** None

### Sprint 10: Integration Hardening & Production Quality

- [ ] **T-027** | P1 | 8 pts | Sprint 10
  **Title:** End-to-end integration testing
  **Description:** Validate complete pipeline: Quarry processes docs → Forge creates curriculum → Foundry trains LoRA → System produces accurate responses. Test with real military technical manuals.
  **Acceptance Criteria:**
  - [ ] Full pipeline runs without manual intervention
  - [ ] Multiple document types processed successfully
  - [ ] Curriculum from Sprint 7 trains successfully
  - [ ] Retrieval + generation accuracy meets targets (>80%)
  - [ ] Performance acceptable on target hardware
  **Files:** tests/integration/test_end_to_end.py
  **Depends On:** T-024
  **Blocked By:** None

- [ ] **T-028** | P1 | 8 pts | Sprint 10
  **Title:** Production hardening and edge case handling
  **Description:** Handle edge cases, improve error messages, add retry logic, optimize performance. Make system robust for real use.
  **Acceptance Criteria:**
  - [ ] Graceful failure modes for all components
  - [ ] Clear error messages (not stack traces)
  - [ ] Retry logic for transient failures
  - [ ] Performance optimizations applied
  - [ ] Resource usage monitored and limited
  **Files:** (across all modules)
  **Depends On:** T-027
  **Blocked By:** None

- [ ] **T-029** | P2 | 5 pts | Sprint 10
  **Title:** Documentation and deployment guide
  **Description:** Complete documentation for all three phases. Installation guide, architecture docs, troubleshooting, export format specs.
  **Acceptance Criteria:**
  - [ ] Installation guide tested on clean system
  - [ ] Architecture documentation complete
  - [ ] API documentation generated
  - [ ] Troubleshooting guide written
  - [ ] Export format specs finalized
  **Files:** docs/
  **Depends On:** T-028
  **Blocked By:** None

**MILESTONE:** ✓ Integrated Pipeline Complete
- Quarry → Forge → Foundry working end-to-end
- Accurate domain responses from LoRA + RAG
- Production-quality and hardened
- System handles edge cases gracefully

---

## PHASE 4: HEARTH + MVP PACKAGE (Sprint 11, 2 weeks)

### Sprint 11: Hearth Interface & Feedback Loop

- [ ] **T-030** | P1 | 13 pts | Sprint 11
  **Title:** Build Hearth chat interface within Kiln
  **Description:** Create chat UI where practitioners interact with trained models. Model switching, document browsing, citation display. Integrated view within Kiln, not standalone.
  **Acceptance Criteria:**
  - [ ] Chat interface functional
  - [ ] Model switching (select LoRA, base model)
  - [ ] Query → response → citations working
  - [ ] Document browser for knowledge base
  - [ ] Response quality feedback buttons
  - [ ] Multi-discipline query mode (dual-query)
  **Files:** ui/src/components/Hearth/, hearth/src/inference.py
  **Depends On:** T-024
  **Blocked By:** None

- [ ] **T-031** | P1 | 8 pts | Sprint 11
  **Title:** Implement feedback capture and routing
  **Description:** Capture interaction signals (accepted answers, follow-ups, rephrased queries, flagged errors). Route to appropriate improvement workflows (Quarry or Forge).
  **Acceptance Criteria:**
  - [ ] All interaction signals logged
  - [ ] Signal analysis identifies patterns
  - [ ] Routing logic: retrieval failures → Quarry, poor responses → Forge
  - [ ] Discipline owner dashboard shows concrete issues
  - [ ] NO automated training data generation (human authority maintained)
  **Files:** hearth/src/feedback.py, hearth/tests/test_feedback.py
  **Depends On:** T-030
  **Blocked By:** None

- [ ] **T-032** | P1 | 5 pts | Sprint 11
  **Title:** MVP packaging and demonstration
  **Description:** Package complete MVP. Create demonstration video/script. Document "proof of concept" validation criteria.
  **Acceptance Criteria:**
  - [ ] All four tools functional in unified Kiln interface
  - [ ] End-to-end demo script written
  - [ ] Military maintenance discipline working example
  - [ ] Performance benchmarks documented
  - [ ] Known limitations clearly stated
  - [ ] Post-MVP roadmap defined
  **Files:** docs/MVP_DEMO.md, docs/VALIDATION_CRITERIA.md
  **Depends On:** T-031
  **Blocked By:** None

**MILESTONE:** ✓✓✓ KILN MVP COMPLETE ✓✓✓
- Complete integrated platform operational
- Quarry processes documents
- Forge builds curricula
- Foundry trains and evaluates models
- Hearth enables interaction
- Feedback flows back for improvement
- **The concept is proven end-to-end**

---

## In Progress
<!-- Tasks currently being worked on -->

## In Review
<!-- Tasks with implementation done, awaiting security + QA review -->

## Done
<!-- Completed tasks this sprint -->

## Icebox
<!-- Unscheduled work: tech debt, nice-to-haves, future ideas -->

### Post-MVP Priorities
- [ ] OCR support in Quarry (scanned/image PDFs)
- [ ] Forge facilitator model training
- [ ] Multi-discipline support (simultaneous)
- [ ] Second discipline validation (non-military)
- [ ] Expanded classifier corpus
- [ ] Subscription infrastructure
- [ ] Additional vector database integrations
- [ ] Community structural profile contributions

---

## Conflict Map

| Branch | Files Modified | Overlaps With |
|--------|---------------|---------------|
| (none active) | — | — |

## Sprint Metrics

- **Current Sprint:** Not started
- **Velocity:** 0 story points
- **Tasks Completed:** 0 / 32
- **Average Cycle Time:** —
- **Quality Gate Failures:** 0
- **Conflicts Prevented:** 0

## Notes

**MVP is not the product.** It is proof that the product works. Everything after MVP builds on validated foundations rather than assumptions.

**Conservative timeline.** 15-20 hours/week part-time. Additional developers would compress timeline.

**Quarry head start.** ~70% complete means Phase 1 compressed vs. greenfield build.

**Focus discipline.** Military maintenance for MVP. Architecture proven cross-discipline.
