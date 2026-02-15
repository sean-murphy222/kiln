# Foundry

Model training, evaluation, and versioning for Kiln. Manages complete lifecycle of discipline-specific LoRA adapters.

## Status

Not started. Phase 3 of MVP roadmap (Sprints 8-10).

## Training Pipeline

**Input:** Forge curriculum (JSONL) + selected base model
**Process:** LoRA configuration with sensible defaults
**Output:** Trained adapter (10-100MB)

Uses standard tooling (Unsloth or Axolotl) under the hood. Advanced settings available but defaults tuned for Forge output.

**Base model selection guidance:**
- Deployment hardware constraints
- Discipline characteristics from Forge curriculum
- Recommended models: Phi, Llama, Mistral, Qwen (3-20B parameters)

## Automated Competency-Based Evaluation

**Three evaluation layers:**

1. **Competency Testing**
   - Held-out test set (reserved during Forge curriculum building)
   - Results per competency area: "Procedural comprehension: 9/10 correct"
   - Immediately actionable

2. **Comparative Evaluation**
   - Same queries through base model without LoRA
   - Side-by-side comparison shows training effectiveness

3. **RAG-Integrated Evaluation**
   - End-to-end queries requiring discipline understanding + retrieval
   - "24/30 answered correctly with citations, 4 partial, 2 incorrect"
   - Links to specific query/response for review

**Language:** Same as curriculum. No loss curves, perplexity, confusion matrices.

## Regression Testing

- Evaluation suite stored with timestamp + triggering event
- Version history per competency (green/yellow/red indicators)
- Auto-triggered on: retrain, LoRA merge, base model swap, Quarry reprocessing
- Rollback to any previous version if regression detected

## Failure Detection

When results poor despite good curriculum:
- Auto-detect training issues (loss not converging, overfitting)
- Plain-language guidance: "Model memorizing examples vs. learning patterns"
- Actionable: "Add more diverse examples to weak competency areas"

## Model Merging

**Optional:** Combine multiple discipline LoRAs
- Linear or TIES merging techniques
- Fast (minutes, no retraining)
- Auto-evaluated against both source test suites
- Transparent tradeoff: merged accuracy vs. individual model accuracy

## MVP Goal

- Operational training pipeline
- Automated evaluation suite
- Integrated pipeline producing domain responses from military maintenance curriculum
- Production-quality hardening
