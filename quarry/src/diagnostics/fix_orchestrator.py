"""
Fix orchestrator for planning and executing chunk fixes.

Handles:
- Fix planning (order of operations, conflict resolution)
- Fix execution (merge/split chunks)
- Validation (verify improvements)
- Rollback (if fixes make things worse)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import Chunk, ChunkMetadata
from chonk.diagnostics.analyzer import ChunkProblem, DiagnosticAnalyzer
from chonk.diagnostics.fix_strategies import FixAction, FixResult, find_fix_strategy
from chonk.utils.tokens import count_tokens


@dataclass
class FixPlan:
    """A plan for fixing multiple problems."""

    actions: list[FixAction]
    estimated_improvement: float  # 0-1, estimated problem reduction
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "estimated_improvement": self.estimated_improvement,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "total_actions": len(self.actions),
        }


class FixOrchestrator:
    """Orchestrate automatic fixes for chunk problems."""

    def __init__(self):
        self.analyzer = DiagnosticAnalyzer()

    def plan_fixes(
        self,
        problems: list[ChunkProblem],
        chunks: list[Chunk],
        auto_resolve_conflicts: bool = True,
    ) -> FixPlan:
        """
        Create a fix plan for the given problems.

        1. Find fix actions for each problem
        2. Detect conflicts (e.g., merge A+B and split B)
        3. Prioritize actions (high confidence first)
        4. Resolve conflicts (if auto_resolve=True)
        """

        # Step 1: Generate fix actions
        actions = []
        for problem in problems:
            action = find_fix_strategy(problem, chunks)
            if action:
                actions.append(action)

        if not actions:
            return FixPlan(
                actions=[],
                estimated_improvement=0.0,
                warnings=["No automatic fixes available for detected problems"],
            )

        # Step 2: Detect conflicts
        conflicts = self._detect_conflicts(actions)

        # Step 3: Resolve conflicts
        if auto_resolve_conflicts and conflicts:
            actions = self._resolve_conflicts(actions, conflicts)
            conflicts = []  # Cleared after resolution

        # Step 4: Sort by confidence (high first) and handle dependencies
        actions = self._sort_and_order(actions, chunks)

        # Step 5: Estimate improvement
        estimated_improvement = min(1.0, len(actions) / max(len(problems), 1) * 0.7)

        warnings = []
        if len(actions) < len(problems):
            warnings.append(f"Only {len(actions)} of {len(problems)} problems have automatic fixes")

        return FixPlan(
            actions=actions,
            estimated_improvement=estimated_improvement,
            conflicts=conflicts,
            warnings=warnings,
        )

    def execute_plan(
        self,
        plan: FixPlan,
        chunks: list[Chunk],
        validate: bool = True,
    ) -> FixResult:
        """
        Execute the fix plan.

        1. Apply each action in order
        2. Track chunk ID changes
        3. Optionally validate improvements
        """

        if not plan.actions:
            return FixResult(
                success=False,
                chunks_before=len(chunks),
                chunks_after=len(chunks),
                actions_applied=[],
                new_chunks=chunks,
                errors=["No actions in plan"],
            )

        original_count = len(chunks)
        current_chunks = chunks.copy()
        applied_actions = []
        errors = []

        # Execute each action
        for action in plan.actions:
            try:
                if action.action_type == "merge":
                    current_chunks = self._apply_merge(action, current_chunks)
                    applied_actions.append(action)

                elif action.action_type == "split":
                    current_chunks = self._apply_split(action, current_chunks)
                    applied_actions.append(action)

                else:
                    errors.append(f"Unknown action type: {action.action_type}")

            except Exception as e:
                errors.append(f"Failed to apply {action.action_type}: {str(e)}")
                continue

        # Validate improvements if requested
        success = True
        if validate and not errors:
            # Re-run diagnostics on fixed chunks
            from chonk.core.document import ChonkDocument, DocumentMetadata
            from pathlib import Path

            # Create temporary document for validation
            temp_doc = ChonkDocument(
                id="temp_validation",
                source_path=Path("temp"),
                source_type="pdf",
                blocks=[],
                chunks=current_chunks,
                metadata=DocumentMetadata(),
                loader_used="temp",
            )

            new_problems = self.analyzer.analyze_document(temp_doc)

            # Check if problems reduced (should reduce by at least 50% of applied actions)
            expected_reduction = len(applied_actions) * 0.5
            # Count problems associated with chunks we fixed
            old_problem_count = len([p for p in applied_actions])  # Use applied actions as proxy

            # If new problems didn't reduce at all, that's a failure
            # But some problems may remain because one chunk can have multiple problems
            if len(new_problems) > old_problem_count and len(applied_actions) > 0:
                # Log warning but don't fail - partial improvement is still progress
                errors.append(
                    f"Validation: Limited improvement - {len(new_problems)} problems remain after {len(applied_actions)} fixes"
                )
                # Don't set success = False - partial fixes are okay

        return FixResult(
            success=success and not errors,
            chunks_before=original_count,
            chunks_after=len(current_chunks),
            actions_applied=applied_actions,
            new_chunks=current_chunks,
            errors=errors,
        )

    def _detect_conflicts(self, actions: list[FixAction]) -> list[str]:
        """Detect conflicting fix actions."""
        conflicts = []

        # Build chunk usage map
        chunk_actions = {}  # chunk_id -> list of actions
        for action in actions:
            for chunk_id in action.chunk_ids:
                if chunk_id not in chunk_actions:
                    chunk_actions[chunk_id] = []
                chunk_actions[chunk_id].append(action)

        # Find chunks involved in multiple actions
        for chunk_id, chunk_action_list in chunk_actions.items():
            if len(chunk_action_list) > 1:
                action_types = [a.action_type for a in chunk_action_list]
                conflicts.append(
                    f"Chunk {chunk_id} involved in multiple actions: {action_types}"
                )

        return conflicts

    def _resolve_conflicts(
        self,
        actions: list[FixAction],
        conflicts: list[str]
    ) -> list[FixAction]:
        """
        Resolve conflicts by keeping higher-confidence actions.

        For chunks involved in multiple actions:
        - Keep the action with highest confidence
        - If same confidence, prefer merges over splits (safer)
        """

        # Build chunk usage map with actions
        chunk_actions = {}
        for action in actions:
            for chunk_id in action.chunk_ids:
                if chunk_id not in chunk_actions:
                    chunk_actions[chunk_id] = []
                chunk_actions[chunk_id].append(action)

        # Find conflicting chunks and resolve
        actions_to_remove = []  # Track actions to remove

        for chunk_id, chunk_action_list in chunk_actions.items():
            if len(chunk_action_list) > 1:
                # Sort by confidence (high first), then prefer merge
                sorted_actions = sorted(
                    chunk_action_list,
                    key=lambda a: (a.confidence, 1 if a.action_type == "merge" else 0),
                    reverse=True,
                )

                # Keep only the best action, remove others
                for action in sorted_actions[1:]:
                    if action not in actions_to_remove:
                        actions_to_remove.append(action)

        # Return actions excluding the ones to remove
        return [a for a in actions if a not in actions_to_remove]

    def _sort_and_order(
        self,
        actions: list[FixAction],
        chunks: list[Chunk],
    ) -> list[FixAction]:
        """
        Sort actions for optimal execution order.

        Rules:
        1. High confidence actions first
        2. Merges before splits (merges are simpler)
        3. Process chunks from end to start (preserve indices)
        """

        # Build chunk index map
        chunk_index = {chunk.id: i for i, chunk in enumerate(chunks)}

        def sort_key(action: FixAction) -> tuple:
            # Primary: confidence (high first)
            conf = -action.confidence

            # Secondary: action type (merge before split)
            type_order = 0 if action.action_type == "merge" else 1

            # Tertiary: chunk position (later chunks first to preserve indices)
            try:
                first_chunk_idx = chunk_index.get(action.chunk_ids[0], 0)
                pos = -first_chunk_idx
            except (IndexError, KeyError):
                pos = 0

            return (conf, type_order, pos)

        return sorted(actions, key=sort_key)

    def _apply_merge(self, action: FixAction, chunks: list[Chunk]) -> list[Chunk]:
        """Merge specified chunks."""
        if len(action.chunk_ids) < 2:
            raise ValueError("Merge requires at least 2 chunks")

        # Find chunks to merge
        to_merge = []
        chunk_indices = []

        for chunk_id in action.chunk_ids:
            for i, chunk in enumerate(chunks):
                if chunk.id == chunk_id:
                    to_merge.append(chunk)
                    chunk_indices.append(i)
                    break

        if len(to_merge) != len(action.chunk_ids):
            raise ValueError(f"Could not find all chunks to merge: {action.chunk_ids}")

        # Sort by index to merge in order
        sorted_pairs = sorted(zip(chunk_indices, to_merge), key=lambda x: x[0])
        sorted_chunks = [chunk for _, chunk in sorted_pairs]

        # Create merged chunk
        merged_content = "\n\n".join(c.content for c in sorted_chunks)
        merged_block_ids = []
        for c in sorted_chunks:
            merged_block_ids.extend(c.block_ids)

        # Combine hierarchy paths
        hierarchy_paths = [c.hierarchy_path for c in sorted_chunks if c.hierarchy_path]
        merged_hierarchy = hierarchy_paths[0] if hierarchy_paths else ""

        merged_chunk = Chunk(
            id=Chunk.generate_id(),
            block_ids=merged_block_ids,
            content=merged_content,
            token_count=count_tokens(merged_content),
            hierarchy_path=merged_hierarchy,
            is_modified=True,
            system_metadata={
                "merged_from": action.chunk_ids,
                "fix_applied": action.description,
            },
            user_metadata=ChunkMetadata(),
        )

        # Remove old chunks and insert merged chunk
        new_chunks = [c for c in chunks if c.id not in action.chunk_ids]
        insert_idx = min(chunk_indices)
        new_chunks.insert(insert_idx, merged_chunk)

        return new_chunks

    def _apply_split(self, action: FixAction, chunks: list[Chunk]) -> list[Chunk]:
        """Split specified chunk."""
        if len(action.chunk_ids) != 1:
            raise ValueError("Split requires exactly 1 chunk")

        chunk_id = action.chunk_ids[0]

        # Find chunk to split
        chunk = None
        chunk_idx = None
        for i, c in enumerate(chunks):
            if c.id == chunk_id:
                chunk = c
                chunk_idx = i
                break

        if chunk is None:
            raise ValueError(f"Could not find chunk to split: {chunk_id}")

        # Determine split point from metadata
        split_type = action.metadata.get("split_type")

        if split_type == "paragraph":
            split_idx = action.metadata.get("split_index", 1)
            paragraphs = chunk.content.split('\n\n')
            content_a = '\n\n'.join(paragraphs[:split_idx])
            content_b = '\n\n'.join(paragraphs[split_idx:])

        elif split_type in ["heading", "midpoint"]:
            split_pos = action.metadata.get("split_position", len(chunk.content) // 2)
            content_a = chunk.content[:split_pos].strip()
            content_b = chunk.content[split_pos:].strip()

        else:
            # Default: split at midpoint
            mid = len(chunk.content) // 2
            content_a = chunk.content[:mid].strip()
            content_b = chunk.content[mid:].strip()

        # Create two new chunks
        chunk_a = Chunk(
            id=Chunk.generate_id(),
            block_ids=chunk.block_ids,  # Both reference same blocks
            content=content_a,
            token_count=count_tokens(content_a),
            hierarchy_path=chunk.hierarchy_path,
            is_modified=True,
            system_metadata={
                "split_from": chunk_id,
                "split_part": "A",
                "fix_applied": action.description,
            },
            user_metadata=ChunkMetadata(),
        )

        chunk_b = Chunk(
            id=Chunk.generate_id(),
            block_ids=chunk.block_ids,
            content=content_b,
            token_count=count_tokens(content_b),
            hierarchy_path=chunk.hierarchy_path,
            is_modified=True,
            system_metadata={
                "split_from": chunk_id,
                "split_part": "B",
                "fix_applied": action.description,
            },
            user_metadata=ChunkMetadata(),
        )

        # Replace original chunk with two new chunks
        new_chunks = chunks[:chunk_idx] + [chunk_a, chunk_b] + chunks[chunk_idx + 1:]

        return new_chunks
