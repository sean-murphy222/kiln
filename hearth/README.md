# Hearth

Interaction and feedback layer for Kiln. Built-in chat interface where practitioners use trained models with Quarry knowledge bases.

## Status

Not started. Phase 4 of MVP roadmap (Sprint 11).

## Architecture

**Hearth is a view within Kiln**, not standalone deployment. Eliminates deployment packaging complexity.

## Interaction

- Chat interface with citations to source documents
- Document browser for processed knowledge base
- Model switching (discipline-specific LoRAs or merged models)
- Immediate switching, no restart required

## Multi-Discipline Queries

**Dual-query mode:**
- Question routed to multiple discipline models simultaneously
- Receive answers from different lenses
- Comparison summary: overlaps and divergences
- User selects which disciplines to query (MVP)
- Tradeoff: Multi-discipline queries take longer (sequential inference on local hardware)

## Feedback Capture

**Signals collected:**
- Accepted answers → implicit positive
- Follow-up questions → coverage gaps
- Rephrased questions → retrieval/comprehension issues
- Flagged incorrect answers → strong negative

**Signal routing:**
- Consistent retrieval failures → Quarry processing issues
- Poor responses with good chunks → Forge training gaps

## Discipline Owner Dashboard

**Surfaces concrete information:**
- Specific questions that struggled
- Weak competency areas
- Document sections retrieval consistently misses

**Human authority maintained:**
- Training data changes NEVER automated
- System surfaces opportunities
- Human decides whether to act
- No synthetic data from user interactions

## Local-First Operation

- Runs entirely on organization's hardware
- Laptop: quantized 7-8B model
- Workstation with modest GPU: 20B model
- No cloud connectivity required
- Zero inference costs after setup

## MVP Goal

- Operational chat interface within Kiln
- Model switching functional
- Feedback capture implemented
- Complete closed loop: Quarry → Forge → Foundry → Hearth → feedback → improvements
