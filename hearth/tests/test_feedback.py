"""Tests for feedback capture and routing system.

Verifies signal capture, routing logic, pattern analysis,
dashboard generation, and the critical human authority principle.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from hearth.src.feedback import (
    DashboardSummary,
    FeedbackError,
    FeedbackManager,
    FeedbackPattern,
    FeedbackSignal,
    FeedbackStore,
    PatternAnalyzer,
    Priority,
    RoutingDecision,
    RoutingTarget,
    SignalRouter,
    SignalType,
)

# ===================================================================
# Fixtures
# ===================================================================


def _make_signal(
    signal_type: SignalType = SignalType.ACCEPTED,
    conversation_id: str = "conv_001",
    query: str = "How do I replace a turbine blade?",
    response: str | None = "Follow TM 1-2840-248-23, Section 5-12...",
    user_comment: str | None = None,
    discipline_id: str | None = "disc_maint001",
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    signal_id: str | None = None,
) -> FeedbackSignal:
    """Create a FeedbackSignal for testing."""
    return FeedbackSignal(
        signal_id=signal_id or "sig_test001",
        signal_type=signal_type,
        conversation_id=conversation_id,
        query=query,
        response=response,
        user_comment=user_comment,
        discipline_id=discipline_id,
        timestamp=timestamp or datetime(2026, 2, 20, 10, 0, 0),
        metadata=metadata or {},
    )


@pytest.fixture()
def store() -> FeedbackStore:
    """Provide a clean FeedbackStore."""
    return FeedbackStore()


@pytest.fixture()
def router() -> SignalRouter:
    """Provide a SignalRouter."""
    return SignalRouter()


@pytest.fixture()
def analyzer() -> PatternAnalyzer:
    """Provide a PatternAnalyzer with low threshold for testing."""
    return PatternAnalyzer(min_signals=2)


@pytest.fixture()
def manager() -> FeedbackManager:
    """Provide a FeedbackManager with default components."""
    return FeedbackManager()


# ===================================================================
# TestFeedbackSignal
# ===================================================================


class TestFeedbackSignal:
    """Test FeedbackSignal dataclass construction and properties."""

    def test_create_signal_all_fields(self) -> None:
        """Signal stores all provided fields."""
        ts = datetime(2026, 2, 20, 10, 0, 0)
        signal = FeedbackSignal(
            signal_id="sig_abc123",
            signal_type=SignalType.ACCEPTED,
            conversation_id="conv_001",
            query="What is procedure for oil change?",
            response="Follow TM section 3-4...",
            user_comment="Helpful answer",
            discipline_id="disc_maint001",
            timestamp=ts,
            metadata={"model": "llama-7b", "citations": ["chunk_01"]},
        )
        assert signal.signal_id == "sig_abc123"
        assert signal.signal_type == SignalType.ACCEPTED
        assert signal.conversation_id == "conv_001"
        assert signal.query == "What is procedure for oil change?"
        assert signal.response == "Follow TM section 3-4..."
        assert signal.user_comment == "Helpful answer"
        assert signal.discipline_id == "disc_maint001"
        assert signal.timestamp == ts
        assert signal.metadata["model"] == "llama-7b"

    def test_create_signal_optional_none(self) -> None:
        """Signal works with optional fields set to None."""
        signal = _make_signal(
            response=None,
            user_comment=None,
            discipline_id=None,
        )
        assert signal.response is None
        assert signal.user_comment is None
        assert signal.discipline_id is None

    def test_all_signal_types_exist(self) -> None:
        """All expected signal types are defined."""
        expected = {
            "accepted",
            "rejected",
            "follow_up",
            "rephrased",
            "flagged_error",
            "citation_clicked",
            "no_result",
        }
        actual = {st.value for st in SignalType}
        assert actual == expected

    def test_signal_to_dict(self) -> None:
        """Signal serializes to dictionary."""
        signal = _make_signal()
        result = signal.to_dict()
        assert result["signal_id"] == "sig_test001"
        assert result["signal_type"] == "accepted"
        assert result["conversation_id"] == "conv_001"
        assert isinstance(result["timestamp"], str)

    def test_signal_from_dict_roundtrip(self) -> None:
        """Signal survives to_dict/from_dict roundtrip."""
        original = _make_signal(
            metadata={"model": "llama-7b"},
            user_comment="Great answer",
        )
        data = original.to_dict()
        restored = FeedbackSignal.from_dict(data)
        assert restored.signal_id == original.signal_id
        assert restored.signal_type == original.signal_type
        assert restored.query == original.query
        assert restored.metadata == original.metadata
        assert restored.user_comment == original.user_comment

    def test_signal_empty_metadata(self) -> None:
        """Signal works with empty metadata dict."""
        signal = _make_signal(metadata={})
        assert signal.metadata == {}


# ===================================================================
# TestFeedbackStore
# ===================================================================


class TestFeedbackStore:
    """Test in-memory signal storage."""

    def test_record_and_retrieve(self, store: FeedbackStore) -> None:
        """Recorded signal can be retrieved."""
        signal = _make_signal()
        store.record(signal)
        results = store.get_signals()
        assert len(results) == 1
        assert results[0].signal_id == "sig_test001"

    def test_record_multiple(self, store: FeedbackStore) -> None:
        """Multiple signals are stored independently."""
        for i in range(5):
            store.record(_make_signal(signal_id=f"sig_{i:03d}"))
        assert len(store.get_signals()) == 5

    def test_get_by_conversation(self, store: FeedbackStore) -> None:
        """Filter signals by conversation ID."""
        store.record(_make_signal(conversation_id="conv_A", signal_id="sig_a"))
        store.record(_make_signal(conversation_id="conv_B", signal_id="sig_b"))
        store.record(_make_signal(conversation_id="conv_A", signal_id="sig_c"))

        results = store.get_signals(conversation_id="conv_A")
        assert len(results) == 2
        assert all(s.conversation_id == "conv_A" for s in results)

    def test_get_by_type(self, store: FeedbackStore) -> None:
        """Filter signals by signal type."""
        store.record(_make_signal(signal_type=SignalType.ACCEPTED, signal_id="s1"))
        store.record(_make_signal(signal_type=SignalType.REJECTED, signal_id="s2"))
        store.record(_make_signal(signal_type=SignalType.ACCEPTED, signal_id="s3"))

        results = store.get_signals_by_type(SignalType.REJECTED)
        assert len(results) == 1
        assert results[0].signal_id == "s2"

    def test_get_by_discipline(self, store: FeedbackStore) -> None:
        """Filter signals by discipline ID."""
        store.record(_make_signal(discipline_id="disc_A", signal_id="s1"))
        store.record(_make_signal(discipline_id="disc_B", signal_id="s2"))
        store.record(_make_signal(discipline_id="disc_A", signal_id="s3"))

        results = store.get_signals(discipline_id="disc_A")
        assert len(results) == 2

    def test_get_in_date_range(self, store: FeedbackStore) -> None:
        """Filter signals within a date range."""
        base = datetime(2026, 2, 20, 10, 0, 0)
        store.record(_make_signal(timestamp=base - timedelta(days=5), signal_id="old"))
        store.record(_make_signal(timestamp=base, signal_id="mid"))
        store.record(_make_signal(timestamp=base + timedelta(days=5), signal_id="new"))

        results = store.get_signals_in_range(
            start=base - timedelta(days=1),
            end=base + timedelta(days=1),
        )
        assert len(results) == 1
        assert results[0].signal_id == "mid"

    def test_count_by_type(self, store: FeedbackStore) -> None:
        """Count signals grouped by type."""
        store.record(_make_signal(signal_type=SignalType.ACCEPTED, signal_id="s1"))
        store.record(_make_signal(signal_type=SignalType.ACCEPTED, signal_id="s2"))
        store.record(_make_signal(signal_type=SignalType.REJECTED, signal_id="s3"))
        store.record(_make_signal(signal_type=SignalType.FLAGGED_ERROR, signal_id="s4"))

        counts = store.count_by_type()
        assert counts[SignalType.ACCEPTED] == 2
        assert counts[SignalType.REJECTED] == 1
        assert counts[SignalType.FLAGGED_ERROR] == 1

    def test_count_by_type_empty(self, store: FeedbackStore) -> None:
        """Count returns zeros for all types when empty."""
        counts = store.count_by_type()
        for st in SignalType:
            assert counts[st] == 0

    def test_get_signals_empty_store(self, store: FeedbackStore) -> None:
        """Empty store returns empty list."""
        assert store.get_signals() == []

    def test_record_duplicate_signal_id(self, store: FeedbackStore) -> None:
        """Duplicate signal IDs raise an error."""
        store.record(_make_signal(signal_id="sig_dup"))
        with pytest.raises(FeedbackError, match="already exists"):
            store.record(_make_signal(signal_id="sig_dup"))


# ===================================================================
# TestSignalRouter
# ===================================================================


class TestSignalRouter:
    """Test routing logic for feedback signals."""

    def test_accepted_routes_to_none(self, router: SignalRouter) -> None:
        """Accepted responses need no action."""
        signal = _make_signal(signal_type=SignalType.ACCEPTED)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.NONE

    def test_citation_clicked_routes_to_none(self, router: SignalRouter) -> None:
        """Citation clicks are positive signals, no action needed."""
        signal = _make_signal(signal_type=SignalType.CITATION_CLICKED)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.NONE

    def test_no_result_routes_to_quarry(self, router: SignalRouter) -> None:
        """No results indicate retrieval/indexing issues."""
        signal = _make_signal(signal_type=SignalType.NO_RESULT, response=None)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.QUARRY

    def test_no_result_has_high_priority(self, router: SignalRouter) -> None:
        """No results get at least MEDIUM priority."""
        signal = _make_signal(signal_type=SignalType.NO_RESULT, response=None)
        decision = router.route(signal)
        assert decision.priority in (Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL)

    def test_rephrased_routes_to_quarry(self, router: SignalRouter) -> None:
        """Rephrased queries suggest query understanding issues."""
        signal = _make_signal(signal_type=SignalType.REPHRASED)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.QUARRY

    def test_flagged_error_routes_to_forge(self, router: SignalRouter) -> None:
        """Flagged errors indicate model response quality issues."""
        signal = _make_signal(signal_type=SignalType.FLAGGED_ERROR)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.FORGE

    def test_flagged_error_high_priority(self, router: SignalRouter) -> None:
        """Flagged errors are high priority."""
        signal = _make_signal(signal_type=SignalType.FLAGGED_ERROR)
        decision = router.route(signal)
        assert decision.priority in (Priority.HIGH, Priority.CRITICAL)

    def test_rejected_low_citations_routes_quarry(self, router: SignalRouter) -> None:
        """Rejected with low citation scores -> retrieval issue."""
        signal = _make_signal(
            signal_type=SignalType.REJECTED,
            metadata={"citation_score": 0.2},
        )
        decision = router.route(signal)
        assert decision.target == RoutingTarget.QUARRY

    def test_rejected_good_citations_routes_forge(self, router: SignalRouter) -> None:
        """Rejected with good citations -> model quality issue."""
        signal = _make_signal(
            signal_type=SignalType.REJECTED,
            metadata={"citation_score": 0.9},
        )
        decision = router.route(signal)
        assert decision.target == RoutingTarget.FORGE

    def test_rejected_no_citation_score_routes_forge(self, router: SignalRouter) -> None:
        """Rejected without citation info defaults to Forge."""
        signal = _make_signal(signal_type=SignalType.REJECTED, metadata={})
        decision = router.route(signal)
        assert decision.target == RoutingTarget.FORGE

    def test_follow_up_routes_to_quarry(self, router: SignalRouter) -> None:
        """Follow-up questions suggest incomplete retrieval."""
        signal = _make_signal(signal_type=SignalType.FOLLOW_UP)
        decision = router.route(signal)
        assert decision.target == RoutingTarget.QUARRY

    def test_route_returns_reason(self, router: SignalRouter) -> None:
        """All routing decisions include a human-readable reason."""
        signal = _make_signal(signal_type=SignalType.NO_RESULT)
        decision = router.route(signal)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_route_returns_suggested_action(self, router: SignalRouter) -> None:
        """All routing decisions include a suggested action."""
        signal = _make_signal(signal_type=SignalType.FLAGGED_ERROR)
        decision = router.route(signal)
        assert isinstance(decision.suggested_action, str)
        assert len(decision.suggested_action) > 0

    def test_route_batch(self, router: SignalRouter) -> None:
        """Batch routing processes all signals."""
        signals = [
            _make_signal(signal_type=SignalType.ACCEPTED, signal_id="s1"),
            _make_signal(signal_type=SignalType.NO_RESULT, signal_id="s2"),
            _make_signal(signal_type=SignalType.FLAGGED_ERROR, signal_id="s3"),
        ]
        decisions = router.route_batch(signals)
        assert len(decisions) == 3
        assert decisions[0].target == RoutingTarget.NONE
        assert decisions[1].target == RoutingTarget.QUARRY
        assert decisions[2].target == RoutingTarget.FORGE

    def test_route_batch_empty(self, router: SignalRouter) -> None:
        """Empty batch returns empty list."""
        assert router.route_batch([]) == []

    def test_routing_decision_has_signal_id(self, router: SignalRouter) -> None:
        """Routing decision references the original signal."""
        signal = _make_signal(signal_id="sig_ref_test")
        decision = router.route(signal)
        assert decision.signal_id == "sig_ref_test"


# ===================================================================
# TestRoutingDecision
# ===================================================================


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_create_routing_decision(self) -> None:
        """RoutingDecision stores all fields."""
        decision = RoutingDecision(
            target=RoutingTarget.QUARRY,
            priority=Priority.HIGH,
            reason="No results found for query",
            suggested_action="Review indexing for turbine maintenance documents",
            signal_id="sig_001",
        )
        assert decision.target == RoutingTarget.QUARRY
        assert decision.priority == Priority.HIGH
        assert "No results" in decision.reason
        assert decision.signal_id == "sig_001"

    def test_routing_target_values(self) -> None:
        """All routing targets are defined."""
        expected = {"quarry", "forge", "none"}
        actual = {rt.value for rt in RoutingTarget}
        assert actual == expected

    def test_priority_values(self) -> None:
        """All priority levels are defined."""
        expected = {"low", "medium", "high", "critical"}
        actual = {p.value for p in Priority}
        assert actual == expected

    def test_routing_decision_to_dict(self) -> None:
        """RoutingDecision serializes to dict."""
        decision = RoutingDecision(
            target=RoutingTarget.FORGE,
            priority=Priority.HIGH,
            reason="User flagged an error",
            suggested_action="Review model response quality",
            signal_id="sig_abc",
        )
        data = decision.to_dict()
        assert data["target"] == "forge"
        assert data["priority"] == "high"
        assert data["signal_id"] == "sig_abc"


# ===================================================================
# TestPatternAnalyzer
# ===================================================================


class TestPatternAnalyzer:
    """Test pattern detection from aggregated signals."""

    def test_detect_repeated_failures(self, analyzer: PatternAnalyzer) -> None:
        """Detects when the same query keeps failing."""
        signals = [
            _make_signal(
                signal_type=SignalType.NO_RESULT,
                query="turbine blade replacement",
                signal_id=f"s{i}",
                timestamp=datetime(2026, 2, 20, 10 + i, 0, 0),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert len(patterns) >= 1
        pattern = patterns[0]
        assert pattern.signal_count >= 2
        assert "turbine blade replacement" in pattern.affected_queries

    def test_detect_low_acceptance_area(self, analyzer: PatternAnalyzer) -> None:
        """Detects discipline areas with low acceptance rates."""
        signals = [
            _make_signal(
                signal_type=SignalType.REJECTED,
                discipline_id="disc_weak",
                signal_id=f"s{i}",
                query=f"query about topic {i}",
                timestamp=datetime(2026, 2, 20, 10 + i, 0, 0),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert len(patterns) >= 1
        found_low_acceptance = any(p.pattern_type == "low_acceptance" for p in patterns)
        assert found_low_acceptance

    def test_detect_missing_coverage(self, analyzer: PatternAnalyzer) -> None:
        """Detects queries that consistently get no results."""
        signals = [
            _make_signal(
                signal_type=SignalType.NO_RESULT,
                query="hydraulic pump maintenance",
                discipline_id="disc_hydro",
                signal_id=f"s{i}",
                timestamp=datetime(2026, 2, 20, 10 + i, 0, 0),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert len(patterns) >= 1
        found_missing = any(p.pattern_type == "missing_coverage" for p in patterns)
        assert found_missing

    def test_no_patterns_below_threshold(self) -> None:
        """No patterns reported when signals are below threshold."""
        analyzer = PatternAnalyzer(min_signals=10)
        signals = [
            _make_signal(signal_type=SignalType.NO_RESULT, signal_id=f"s{i}") for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert patterns == []

    def test_empty_signals_no_patterns(self, analyzer: PatternAnalyzer) -> None:
        """Empty signal list produces no patterns."""
        assert analyzer.analyze([]) == []

    def test_pattern_has_routing(self, analyzer: PatternAnalyzer) -> None:
        """Each pattern includes a routing decision."""
        signals = [
            _make_signal(
                signal_type=SignalType.FLAGGED_ERROR,
                signal_id=f"s{i}",
                query="same question",
                timestamp=datetime(2026, 2, 20, 10 + i, 0, 0),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert len(patterns) >= 1
        for pattern in patterns:
            assert isinstance(pattern.routing, RoutingDecision)

    def test_pattern_timestamps(self, analyzer: PatternAnalyzer) -> None:
        """Pattern tracks first and last seen timestamps."""
        base = datetime(2026, 2, 20, 10, 0, 0)
        signals = [
            _make_signal(
                signal_type=SignalType.NO_RESULT,
                query="same failing query",
                signal_id=f"s{i}",
                timestamp=base + timedelta(hours=i),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        assert len(patterns) >= 1
        pattern = patterns[0]
        assert pattern.first_seen <= pattern.last_seen


# ===================================================================
# TestFeedbackPattern
# ===================================================================


class TestFeedbackPattern:
    """Test FeedbackPattern dataclass."""

    def test_create_pattern(self) -> None:
        """Pattern stores all fields correctly."""
        routing = RoutingDecision(
            target=RoutingTarget.QUARRY,
            priority=Priority.HIGH,
            reason="Repeated no-result failures",
            suggested_action="Add documents for turbine maintenance",
            signal_id="pattern_001",
        )
        pattern = FeedbackPattern(
            pattern_id="pat_001",
            pattern_type="repeated_failures",
            description="3 queries about turbines returned no results",
            affected_queries=["turbine blade", "turbine maintenance"],
            signal_count=3,
            first_seen=datetime(2026, 2, 18),
            last_seen=datetime(2026, 2, 20),
            routing=routing,
        )
        assert pattern.pattern_id == "pat_001"
        assert pattern.pattern_type == "repeated_failures"
        assert pattern.signal_count == 3
        assert len(pattern.affected_queries) == 2
        assert pattern.routing.target == RoutingTarget.QUARRY

    def test_pattern_to_dict(self) -> None:
        """Pattern serializes to dictionary."""
        routing = RoutingDecision(
            target=RoutingTarget.FORGE,
            priority=Priority.MEDIUM,
            reason="Low acceptance",
            suggested_action="Add training examples",
            signal_id="pat_002",
        )
        pattern = FeedbackPattern(
            pattern_id="pat_002",
            pattern_type="low_acceptance",
            description="Low acceptance rate in hydraulics",
            affected_queries=["hydraulic pump"],
            signal_count=5,
            first_seen=datetime(2026, 2, 15),
            last_seen=datetime(2026, 2, 20),
            routing=routing,
        )
        data = pattern.to_dict()
        assert data["pattern_id"] == "pat_002"
        assert data["signal_count"] == 5
        assert data["routing"]["target"] == "forge"


# ===================================================================
# TestDashboardSummary
# ===================================================================


class TestDashboardSummary:
    """Test dashboard summary generation."""

    def test_create_dashboard(self) -> None:
        """Dashboard stores all fields."""
        dashboard = DashboardSummary(
            discipline_id="disc_maint001",
            total_queries=100,
            acceptance_rate=0.85,
            rejection_rate=0.10,
            flagged_errors=5,
            top_issues=[],
            quarry_actions=[],
            forge_actions=[],
            period_start=datetime(2026, 2, 1),
            period_end=datetime(2026, 2, 28),
        )
        assert dashboard.discipline_id == "disc_maint001"
        assert dashboard.total_queries == 100
        assert dashboard.acceptance_rate == 0.85
        assert dashboard.rejection_rate == 0.10
        assert dashboard.flagged_errors == 5

    def test_dashboard_rates_are_fractions(self) -> None:
        """Rates must be between 0.0 and 1.0."""
        dashboard = DashboardSummary(
            discipline_id="disc_001",
            total_queries=50,
            acceptance_rate=0.7,
            rejection_rate=0.2,
            flagged_errors=2,
            top_issues=[],
            quarry_actions=[],
            forge_actions=[],
            period_start=datetime(2026, 2, 1),
            period_end=datetime(2026, 2, 28),
        )
        assert 0.0 <= dashboard.acceptance_rate <= 1.0
        assert 0.0 <= dashboard.rejection_rate <= 1.0

    def test_dashboard_to_dict(self) -> None:
        """Dashboard serializes to dictionary."""
        dashboard = DashboardSummary(
            discipline_id="disc_001",
            total_queries=10,
            acceptance_rate=0.5,
            rejection_rate=0.3,
            flagged_errors=1,
            top_issues=[],
            quarry_actions=[],
            forge_actions=[],
            period_start=datetime(2026, 2, 1),
            period_end=datetime(2026, 2, 28),
        )
        data = dashboard.to_dict()
        assert data["discipline_id"] == "disc_001"
        assert data["total_queries"] == 10
        assert isinstance(data["period_start"], str)

    def test_dashboard_zero_queries(self) -> None:
        """Dashboard handles zero queries correctly."""
        dashboard = DashboardSummary(
            discipline_id="disc_empty",
            total_queries=0,
            acceptance_rate=0.0,
            rejection_rate=0.0,
            flagged_errors=0,
            top_issues=[],
            quarry_actions=[],
            forge_actions=[],
            period_start=datetime(2026, 2, 1),
            period_end=datetime(2026, 2, 28),
        )
        assert dashboard.total_queries == 0
        assert dashboard.acceptance_rate == 0.0


# ===================================================================
# TestFeedbackManager
# ===================================================================


class TestFeedbackManager:
    """Test the main orchestrator for feedback capture and routing."""

    def test_capture_creates_signal(self, manager: FeedbackManager) -> None:
        """Capture returns a well-formed signal."""
        signal = manager.capture(
            signal_type=SignalType.ACCEPTED,
            conversation_id="conv_001",
            query="How to replace turbine blade?",
            response="Follow TM section 5-12...",
        )
        assert isinstance(signal, FeedbackSignal)
        assert signal.signal_type == SignalType.ACCEPTED
        assert signal.conversation_id == "conv_001"
        assert len(signal.signal_id) > 0

    def test_capture_stores_signal(self, manager: FeedbackManager) -> None:
        """Captured signal is stored and retrievable."""
        signal = manager.capture(
            signal_type=SignalType.REJECTED,
            conversation_id="conv_002",
            query="What is the torque spec?",
        )
        stored = manager.store.get_signals(conversation_id="conv_002")
        assert len(stored) == 1
        assert stored[0].signal_id == signal.signal_id

    def test_capture_generates_unique_ids(self, manager: FeedbackManager) -> None:
        """Each captured signal gets a unique ID."""
        ids = set()
        for _ in range(20):
            signal = manager.capture(
                signal_type=SignalType.ACCEPTED,
                conversation_id="conv_bulk",
                query="test query",
            )
            ids.add(signal.signal_id)
        assert len(ids) == 20

    def test_get_routing_for_signal(self, manager: FeedbackManager) -> None:
        """Can retrieve routing decision for a captured signal."""
        signal = manager.capture(
            signal_type=SignalType.NO_RESULT,
            conversation_id="conv_003",
            query="hydraulic pump spec",
        )
        decision = manager.get_routing(signal.signal_id)
        assert isinstance(decision, RoutingDecision)
        assert decision.target == RoutingTarget.QUARRY

    def test_get_routing_unknown_signal_raises(self, manager: FeedbackManager) -> None:
        """Requesting routing for unknown signal raises error."""
        with pytest.raises(FeedbackError, match="not found"):
            manager.get_routing("sig_nonexistent")

    def test_get_dashboard(self, manager: FeedbackManager) -> None:
        """Dashboard aggregates signals for a discipline."""
        for i in range(5):
            manager.capture(
                signal_type=SignalType.ACCEPTED,
                conversation_id=f"conv_{i}",
                query=f"question {i}",
                discipline_id="disc_maint001",
            )
        for i in range(2):
            manager.capture(
                signal_type=SignalType.REJECTED,
                conversation_id=f"conv_rej_{i}",
                query=f"bad question {i}",
                discipline_id="disc_maint001",
            )
        manager.capture(
            signal_type=SignalType.FLAGGED_ERROR,
            conversation_id="conv_err",
            query="wrong answer",
            discipline_id="disc_maint001",
        )

        dashboard = manager.get_dashboard("disc_maint001", days=30)
        assert isinstance(dashboard, DashboardSummary)
        assert dashboard.total_queries == 8
        assert dashboard.flagged_errors == 1
        assert dashboard.acceptance_rate > 0.0
        assert dashboard.rejection_rate > 0.0

    def test_get_dashboard_no_signals(self, manager: FeedbackManager) -> None:
        """Dashboard for discipline with no signals returns zeros."""
        dashboard = manager.get_dashboard("disc_none", days=30)
        assert dashboard.total_queries == 0
        assert dashboard.acceptance_rate == 0.0
        assert dashboard.rejection_rate == 0.0

    def test_get_patterns(self, manager: FeedbackManager) -> None:
        """Pattern detection works through the manager."""
        for i in range(5):
            manager.capture(
                signal_type=SignalType.NO_RESULT,
                conversation_id=f"conv_{i}",
                query="same failing query",
                discipline_id="disc_maint001",
            )
        patterns = manager.get_patterns(discipline_id="disc_maint001")
        assert len(patterns) >= 1

    def test_export_signals(self, manager: FeedbackManager, tmp_path: Path) -> None:
        """Signals export to JSONL file."""
        for i in range(3):
            manager.capture(
                signal_type=SignalType.ACCEPTED,
                conversation_id=f"conv_{i}",
                query=f"query {i}",
            )
        export_path = tmp_path / "signals.jsonl"
        count = manager.export_signals(export_path)
        assert count == 3
        assert export_path.exists()

        lines = export_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "signal_id" in data
            assert "signal_type" in data

    def test_export_empty_store(self, manager: FeedbackManager, tmp_path: Path) -> None:
        """Exporting empty store writes empty file, returns 0."""
        export_path = tmp_path / "empty.jsonl"
        count = manager.export_signals(export_path)
        assert count == 0
        assert export_path.exists()
        assert export_path.read_text() == ""

    def test_capture_with_all_optional_fields(self, manager: FeedbackManager) -> None:
        """Capture accepts all optional parameters."""
        signal = manager.capture(
            signal_type=SignalType.FLAGGED_ERROR,
            conversation_id="conv_full",
            query="wrong safety procedure",
            response="Incorrect: skip lockout...",
            user_comment="This is dangerously wrong!",
            discipline_id="disc_safety",
            metadata={"model": "llama-7b", "citations": ["chunk_01", "chunk_02"]},
        )
        assert signal.user_comment == "This is dangerously wrong!"
        assert signal.discipline_id == "disc_safety"
        assert signal.metadata["model"] == "llama-7b"

    def test_manager_uses_custom_components(self) -> None:
        """Manager accepts injected store, router, analyzer."""
        store = FeedbackStore()
        router = SignalRouter()
        analyzer = PatternAnalyzer(min_signals=10)
        mgr = FeedbackManager(store=store, router=router, analyzer=analyzer)
        assert mgr.store is store
        assert mgr.router is router
        assert mgr.analyzer is analyzer


# ===================================================================
# TestHumanAuthority
# ===================================================================


class TestHumanAuthority:
    """Verify that no automated training data generation occurs.

    This is the MOST CRITICAL test class. The feedback system must
    NEVER auto-generate training data, auto-modify Forge curricula,
    or take automated action. All outputs are SUGGESTIONS for humans.
    """

    def test_routing_decisions_are_suggestions_only(self, router: SignalRouter) -> None:
        """Routing decisions contain suggested_action, not execute_action."""
        signal = _make_signal(signal_type=SignalType.FLAGGED_ERROR)
        decision = router.route(signal)
        assert hasattr(decision, "suggested_action")
        assert not hasattr(decision, "execute_action")
        assert not hasattr(decision, "auto_fix")
        assert not hasattr(decision, "auto_generate")

    def test_no_forge_auto_modification_api(self, manager: FeedbackManager) -> None:
        """Manager has no methods that auto-modify Forge data."""
        forbidden_methods = [
            "auto_generate_examples",
            "create_training_data",
            "auto_add_to_forge",
            "generate_synthetic_data",
            "auto_modify_curriculum",
            "auto_create_example",
        ]
        for method_name in forbidden_methods:
            assert not hasattr(manager, method_name), (
                f"FeedbackManager must NOT have '{method_name}' method. "
                "Human authority is non-negotiable."
            )

    def test_no_quarry_auto_modification_api(self, manager: FeedbackManager) -> None:
        """Manager has no methods that auto-modify Quarry data."""
        forbidden_methods = [
            "auto_reindex",
            "auto_modify_chunks",
            "auto_update_embeddings",
        ]
        for method_name in forbidden_methods:
            assert not hasattr(manager, method_name), (
                f"FeedbackManager must NOT have '{method_name}' method. "
                "Human authority is non-negotiable."
            )

    def test_patterns_are_informational(self, analyzer: PatternAnalyzer) -> None:
        """Patterns describe issues but don't auto-fix them."""
        signals = [
            _make_signal(
                signal_type=SignalType.NO_RESULT,
                query="same query",
                signal_id=f"s{i}",
                timestamp=datetime(2026, 2, 20, 10 + i, 0, 0),
            )
            for i in range(3)
        ]
        patterns = analyzer.analyze(signals)
        for pattern in patterns:
            assert not hasattr(pattern, "auto_fix")
            assert not hasattr(pattern, "auto_resolve")
            assert isinstance(pattern.description, str)
            assert isinstance(pattern.routing.suggested_action, str)

    def test_dashboard_contains_suggestions_not_actions(self, manager: FeedbackManager) -> None:
        """Dashboard provides quarry_actions and forge_actions as suggestions."""
        for i in range(3):
            manager.capture(
                signal_type=SignalType.REJECTED,
                conversation_id=f"conv_{i}",
                query=f"query {i}",
                discipline_id="disc_test",
            )
        dashboard = manager.get_dashboard("disc_test", days=30)
        # Actions are RoutingDecisions (suggestions), not executable commands
        for action in dashboard.quarry_actions + dashboard.forge_actions:
            assert isinstance(action, RoutingDecision)
            assert isinstance(action.suggested_action, str)

    def test_export_does_not_create_training_format(
        self, manager: FeedbackManager, tmp_path: Path
    ) -> None:
        """Exported data is signal logs, NOT training data format."""
        manager.capture(
            signal_type=SignalType.ACCEPTED,
            conversation_id="conv_001",
            query="test query",
            response="test response",
        )
        export_path = tmp_path / "export.jsonl"
        manager.export_signals(export_path)

        line = export_path.read_text().strip()
        data = json.loads(line)
        # Must NOT be Alpaca format (instruction/input/output)
        assert "instruction" not in data
        assert "input" not in data
        assert "output" not in data
        # Must be signal format
        assert "signal_id" in data
        assert "signal_type" in data

    def test_signal_type_enum_has_no_auto_fix_value(self) -> None:
        """SignalType enum must not include auto-fix signal types."""
        forbidden = {"auto_fix", "auto_generate", "synthetic", "auto_train"}
        actual_values = {st.value for st in SignalType}
        overlap = actual_values & forbidden
        assert overlap == set(), (
            f"SignalType must NOT contain {overlap}. " "No automated actions allowed."
        )

    def test_routing_target_has_no_auto_value(self) -> None:
        """RoutingTarget must not include automated targets."""
        forbidden = {"auto", "automatic", "synthetic", "auto_forge", "auto_quarry"}
        actual_values = {rt.value for rt in RoutingTarget}
        overlap = actual_values & forbidden
        assert overlap == set(), (
            f"RoutingTarget must NOT contain {overlap}. " "Human authority is non-negotiable."
        )
