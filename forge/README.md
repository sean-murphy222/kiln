# Forge

Discipline curriculum builder for Kiln. Guides domain experts through creating human-validated training data for fine-tuning small language models.

## Status

Not started. Phase 2 of MVP roadmap (Sprints 4-7).

## Core Concept

Train at the **discipline level**, not document level. Discipline = body of knowledge, conventions, reasoning patterns, practices.

Training teaches the model **how the discipline works** (vocabulary, document conventions, reasoning patterns). Specific content supplied at inference time through RAG.

## The Guided Process

**Step 1: Discipline Discovery (45-60 min)**
- Interview domain expert with structured questions
- Surface essential characteristics: document types, core competencies, common questions
- Build discipline model

**Step 2: Competency Mapping (15-20 min)**
- Translate discipline model into competency map
- Categories the model must demonstrate (e.g., procedural comprehension, fault isolation, safety awareness)
- Expert validates and refines

**Step 3: Example Elicitation (2-4 hours across sessions)**
- Create realistic questions and ideal answers per competency
- Tricky variants where inexperience leads to errors
- Tag with competency area, reasoning pattern, metadata
- Monitor coverage, prompt for gaps

**Step 4: Quality Scaffolding**
- Enforce consistency across curriculum
- Flag cross-contributor inconsistencies
- Identify repetitive or narrow examples

## Multi-Contributor Support

- Multiple SMEs contribute to same discipline
- Each owns competency areas matching their specialty
- Discipline lead reviews for consistency
- Distributed burden, improved coverage

## The Facilitator Model

**Bootstrap strategy:**
1. Framework-only (MVP Phase 2) - deterministic templates, forms, dashboards
2. Capture real sessions with experts (5-8 disciplines)
3. Train facilitator LoRA from human interaction data
4. Integrate and iterate (post-MVP)

Facilitator is advisory, never authoritative. Expert always confirms.

## Target Output

300-500 examples covering full range of discipline competencies. JSONL format for Foundry training pipeline.

## MVP Goal

One validated discipline curriculum for military maintenance, created by real domain expert.
