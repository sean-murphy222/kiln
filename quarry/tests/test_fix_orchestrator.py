"""Regression tests for FixOrchestrator.execute_plan.

The "automatic fixes aren't applying" bug: execute_plan applied all merges but
appended a "limited improvement" validation note to `errors`, and
`success = success and not errors` then flipped success to False, so the endpoint
discarded the applied fixes. The note must be a warning; success means we applied
at least one fix.
"""

from __future__ import annotations

from chonk.core.document import Chunk
from chonk.diagnostics.fix_orchestrator import FixOrchestrator, FixPlan
from chonk.diagnostics.fix_strategies import FixAction


def _c(cid: str, content: str, toks: int) -> Chunk:
    return Chunk(id=cid, block_ids=[], content=content, token_count=toks)


def test_execute_plan_succeeds_despite_remaining_problems() -> None:
    # After merging the two tiny chunks, the merged chunk is still small and c3
    # starts lowercase -> 2 problems remain > 1 applied action -> the validation
    # note fires. It must not flip success to False.
    chunks = [
        _c("c1", "Ok.", 2),
        _c("c2", "No.", 2),
        _c("c3", "and the procedure continues with more detail here for completeness.", 30),
    ]
    plan = FixPlan(
        actions=[FixAction("merge", ["c1", "c2"], "merge tiny chunks", 0.9)],
        estimated_improvement=0.5,
    )

    result = FixOrchestrator().execute_plan(plan, chunks, validate=True)

    assert result.success is True  # was False (the bug)
    assert result.errors == []  # validation note no longer pollutes errors
    assert result.chunks_after < result.chunks_before  # the merge actually applied
    assert any("Limited improvement" in w for w in result.warnings)


def test_execute_plan_empty_plan_is_unsuccessful() -> None:
    result = FixOrchestrator().execute_plan(
        FixPlan(actions=[], estimated_improvement=0.0),
        [_c("c1", "hello", 2)],
        validate=True,
    )
    assert result.success is False
