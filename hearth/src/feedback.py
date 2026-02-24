"""Feedback capture and routing for Hearth interactions.

Captures user interaction signals and routes them to appropriate
improvement workflows. CRITICAL: No automated training data generation.
All improvement suggestions require human review and approval.

Routing rules:
    - NO_RESULT -> QUARRY (missing documents or poor indexing)
    - REJECTED + low citation scores -> QUARRY (retrieval quality)
    - REJECTED + good citations -> FORGE (model needs more training data)
    - FLAGGED_ERROR -> FORGE with HIGH priority
    - REPHRASED -> QUARRY with MEDIUM priority (query understanding)
    - FOLLOW_UP -> QUARRY with LOW priority (incomplete retrieval)
    - ACCEPTED -> NONE (positive signal, no action)
    - CITATION_CLICKED -> NONE (positive signal)
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackError(Exception):
    """Base exception for feedback system errors."""


# ===================================================================
# Enums
# ===================================================================


class SignalType(str, Enum):
    """Type of user interaction signal."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FOLLOW_UP = "follow_up"
    REPHRASED = "rephrased"
    FLAGGED_ERROR = "flagged_error"
    CITATION_CLICKED = "citation_clicked"
    NO_RESULT = "no_result"


class RoutingTarget(str, Enum):
    """Where a feedback signal should be routed for improvement."""

    QUARRY = "quarry"
    FORGE = "forge"
    NONE = "none"


class Priority(str, Enum):
    """Priority level for a routing decision."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class FeedbackSignal:
    """A single feedback signal from a user interaction.

    Attributes:
        signal_id: Unique identifier for this signal.
        signal_type: Category of the interaction signal.
        conversation_id: ID of the conversation this signal belongs to.
        query: The user's original query.
        response: The system's response (None if no result).
        user_comment: Optional free-text feedback from the user.
        discipline_id: Which discipline this query relates to.
        timestamp: When this signal was recorded.
        metadata: Extra context (citations, model used, etc.).
    """

    signal_id: str
    signal_type: SignalType
    conversation_id: str
    query: str
    response: str | None
    user_comment: str | None
    discipline_id: str | None
    timestamp: datetime
    metadata: dict[str, Any]

    @staticmethod
    def generate_id() -> str:
        """Generate a unique signal ID.

        Returns:
            A prefixed UUID string like 'sig_abc123def456'.
        """
        return f"sig_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all fields, datetime as ISO string.
        """
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "response": self.response,
            "user_comment": self.user_comment,
            "discipline_id": self.discipline_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackSignal:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with signal fields.

        Returns:
            Reconstructed FeedbackSignal instance.
        """
        return cls(
            signal_id=data["signal_id"],
            signal_type=SignalType(data["signal_type"]),
            conversation_id=data["conversation_id"],
            query=data["query"],
            response=data.get("response"),
            user_comment=data.get("user_comment"),
            discipline_id=data.get("discipline_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingDecision:
    """Where a feedback signal should be routed.

    All routing decisions are SUGGESTIONS for human reviewers.
    They never trigger automated actions.

    Attributes:
        target: Which improvement workflow to suggest.
        priority: How urgently this should be reviewed.
        reason: Human-readable explanation of the routing.
        suggested_action: What the discipline owner should consider doing.
        signal_id: ID of the signal that produced this decision.
    """

    target: RoutingTarget
    priority: Priority
    reason: str
    suggested_action: str
    signal_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all fields, enums as string values.
        """
        return {
            "target": self.target.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "suggested_action": self.suggested_action,
            "signal_id": self.signal_id,
        }


@dataclass
class FeedbackPattern:
    """An identified pattern from aggregated signals.

    Patterns are informational -- they describe issues for human
    review, never auto-fix anything.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        pattern_type: Category (repeated_failures, low_acceptance, missing_coverage).
        description: Human-readable summary of the pattern.
        affected_queries: Queries that contributed to this pattern.
        signal_count: Number of signals in this pattern.
        first_seen: Timestamp of the earliest signal.
        last_seen: Timestamp of the most recent signal.
        routing: Suggested routing for addressing this pattern.
    """

    pattern_id: str
    pattern_type: str
    description: str
    affected_queries: list[str]
    signal_count: int
    first_seen: datetime
    last_seen: datetime
    routing: RoutingDecision

    @staticmethod
    def generate_id() -> str:
        """Generate a unique pattern ID.

        Returns:
            A prefixed UUID string like 'pat_abc123def456'.
        """
        return f"pat_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all fields, nested routing also serialized.
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "affected_queries": self.affected_queries,
            "signal_count": self.signal_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "routing": self.routing.to_dict(),
        }


@dataclass
class DashboardSummary:
    """Summary data for discipline owner dashboard.

    Provides an overview of interaction quality and actionable
    suggestions. All actions are SUGGESTIONS for human review.

    Attributes:
        discipline_id: Which discipline this summary covers.
        total_queries: Total number of queries in the period.
        acceptance_rate: Fraction of queries accepted (0.0-1.0).
        rejection_rate: Fraction of queries rejected (0.0-1.0).
        flagged_errors: Count of explicitly flagged errors.
        top_issues: Most significant patterns detected.
        quarry_actions: Suggested retrieval improvements.
        forge_actions: Suggested training data improvements.
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
    """

    discipline_id: str
    total_queries: int
    acceptance_rate: float
    rejection_rate: float
    flagged_errors: int
    top_issues: list[FeedbackPattern]
    quarry_actions: list[RoutingDecision]
    forge_actions: list[RoutingDecision]
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all fields, nested objects also serialized.
        """
        return {
            "discipline_id": self.discipline_id,
            "total_queries": self.total_queries,
            "acceptance_rate": self.acceptance_rate,
            "rejection_rate": self.rejection_rate,
            "flagged_errors": self.flagged_errors,
            "top_issues": [p.to_dict() for p in self.top_issues],
            "quarry_actions": [a.to_dict() for a in self.quarry_actions],
            "forge_actions": [a.to_dict() for a in self.forge_actions],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


# ===================================================================
# FeedbackStore
# ===================================================================


class FeedbackStore:
    """In-memory feedback signal storage.

    Provides recording and querying of feedback signals. MVP uses
    in-memory storage; production would use persistent backend.
    """

    def __init__(self) -> None:
        """Initialize empty signal store."""
        self._signals: list[FeedbackSignal] = []
        self._signal_ids: set[str] = set()

    def record(self, signal: FeedbackSignal) -> None:
        """Record a feedback signal.

        Args:
            signal: The signal to store.

        Raises:
            FeedbackError: If a signal with the same ID already exists.
        """
        if signal.signal_id in self._signal_ids:
            raise FeedbackError(f"Signal '{signal.signal_id}' already exists")
        self._signals.append(signal)
        self._signal_ids.add(signal.signal_id)

    def get_signals(
        self,
        conversation_id: str | None = None,
        discipline_id: str | None = None,
    ) -> list[FeedbackSignal]:
        """Get signals with optional filters.

        Args:
            conversation_id: Filter by conversation (optional).
            discipline_id: Filter by discipline (optional).

        Returns:
            List of matching signals.
        """
        results = self._signals
        if conversation_id is not None:
            results = [s for s in results if s.conversation_id == conversation_id]
        if discipline_id is not None:
            results = [s for s in results if s.discipline_id == discipline_id]
        return results

    def get_signals_by_type(self, signal_type: SignalType) -> list[FeedbackSignal]:
        """Get all signals of a specific type.

        Args:
            signal_type: The signal type to filter by.

        Returns:
            List of signals matching the type.
        """
        return [s for s in self._signals if s.signal_type == signal_type]

    def get_signals_in_range(self, start: datetime, end: datetime) -> list[FeedbackSignal]:
        """Get signals within a date range (inclusive).

        Args:
            start: Start of the date range.
            end: End of the date range.

        Returns:
            List of signals within the range.
        """
        return [s for s in self._signals if start <= s.timestamp <= end]

    def count_by_type(self) -> dict[SignalType, int]:
        """Count signals grouped by type.

        Returns:
            Dictionary mapping each SignalType to its count.
        """
        counts: dict[SignalType, int] = {st: 0 for st in SignalType}
        for signal in self._signals:
            counts[signal.signal_type] += 1
        return counts


# ===================================================================
# SignalRouter
# ===================================================================


_CITATION_SCORE_THRESHOLD = 0.5


class SignalRouter:
    """Routes feedback signals to appropriate improvement workflows.

    All routing decisions are SUGGESTIONS for human discipline owners.
    No automated actions are ever taken.

    Routing rules:
        - NO_RESULT -> QUARRY (missing documents or poor indexing)
        - REJECTED + low citation scores -> QUARRY (retrieval quality)
        - REJECTED + good citations -> FORGE (model quality)
        - REJECTED + no citation info -> FORGE (default)
        - FLAGGED_ERROR -> FORGE with HIGH priority
        - REPHRASED -> QUARRY with MEDIUM priority
        - FOLLOW_UP -> QUARRY with LOW priority
        - ACCEPTED -> NONE
        - CITATION_CLICKED -> NONE
    """

    def route(self, signal: FeedbackSignal) -> RoutingDecision:
        """Route a single signal to an improvement target.

        Args:
            signal: The feedback signal to route.

        Returns:
            A RoutingDecision with target, priority, and suggestion.
        """
        target = self._determine_target(signal)
        priority = self._assess_priority(signal)
        suggestion = self._generate_suggestion(signal, target)
        reason = self._generate_reason(signal, target)

        return RoutingDecision(
            target=target,
            priority=priority,
            reason=reason,
            suggested_action=suggestion,
            signal_id=signal.signal_id,
        )

    def route_batch(self, signals: list[FeedbackSignal]) -> list[RoutingDecision]:
        """Route multiple signals at once.

        Args:
            signals: List of feedback signals.

        Returns:
            List of routing decisions, one per signal.
        """
        return [self.route(s) for s in signals]

    def _determine_target(self, signal: FeedbackSignal) -> RoutingTarget:
        """Determine which workflow should review this signal.

        Args:
            signal: The feedback signal to analyze.

        Returns:
            The appropriate routing target.
        """
        if signal.signal_type in (SignalType.ACCEPTED, SignalType.CITATION_CLICKED):
            return RoutingTarget.NONE

        if signal.signal_type == SignalType.NO_RESULT:
            return RoutingTarget.QUARRY

        if signal.signal_type == SignalType.REPHRASED:
            return RoutingTarget.QUARRY

        if signal.signal_type == SignalType.FOLLOW_UP:
            return RoutingTarget.QUARRY

        if signal.signal_type == SignalType.FLAGGED_ERROR:
            return RoutingTarget.FORGE

        if signal.signal_type == SignalType.REJECTED:
            return self._route_rejection(signal)

        return RoutingTarget.NONE  # pragma: no cover

    def _route_rejection(self, signal: FeedbackSignal) -> RoutingTarget:
        """Route a rejection based on citation quality metadata.

        Low citation scores suggest retrieval problems (Quarry).
        Good citations but rejection suggests model problems (Forge).

        Args:
            signal: A REJECTED feedback signal.

        Returns:
            QUARRY if poor citations, FORGE otherwise.
        """
        citation_score = signal.metadata.get("citation_score")
        if citation_score is not None and citation_score < _CITATION_SCORE_THRESHOLD:
            return RoutingTarget.QUARRY
        return RoutingTarget.FORGE

    def _assess_priority(self, signal: FeedbackSignal) -> Priority:
        """Assess how urgently this signal needs human attention.

        Args:
            signal: The feedback signal.

        Returns:
            Priority level for the routing decision.
        """
        priority_map: dict[SignalType, Priority] = {
            SignalType.FLAGGED_ERROR: Priority.HIGH,
            SignalType.NO_RESULT: Priority.MEDIUM,
            SignalType.REJECTED: Priority.MEDIUM,
            SignalType.REPHRASED: Priority.MEDIUM,
            SignalType.FOLLOW_UP: Priority.LOW,
            SignalType.ACCEPTED: Priority.LOW,
            SignalType.CITATION_CLICKED: Priority.LOW,
        }
        return priority_map.get(signal.signal_type, Priority.LOW)

    def _generate_suggestion(self, signal: FeedbackSignal, target: RoutingTarget) -> str:
        """Generate a human-readable suggested action.

        Args:
            signal: The feedback signal.
            target: Where this will be routed.

        Returns:
            Actionable suggestion for the discipline owner.
        """
        if target == RoutingTarget.QUARRY:
            return self._quarry_suggestion(signal)
        if target == RoutingTarget.FORGE:
            return self._forge_suggestion(signal)
        return "No action needed. Positive interaction signal."

    def _quarry_suggestion(self, signal: FeedbackSignal) -> str:
        """Generate a Quarry-targeted suggestion.

        Args:
            signal: The feedback signal.

        Returns:
            Suggestion string for retrieval improvement.
        """
        if signal.signal_type == SignalType.NO_RESULT:
            return (
                "Review document indexing for this topic. "
                "Consider adding relevant documents to Quarry."
            )
        if signal.signal_type == SignalType.REPHRASED:
            return (
                "User rephrased their query, suggesting the initial "
                "retrieval did not surface relevant content. "
                "Review query-to-chunk mapping."
            )
        if signal.signal_type == SignalType.FOLLOW_UP:
            return (
                "User asked a follow-up, which may indicate "
                "incomplete retrieval. Review chunk boundaries."
            )
        return (
            "Review retrieval quality for this query. "
            "Citation scores suggest poor document matching."
        )

    def _forge_suggestion(self, signal: FeedbackSignal) -> str:
        """Generate a Forge-targeted suggestion.

        Args:
            signal: The feedback signal.

        Returns:
            Suggestion string for training data improvement.
        """
        if signal.signal_type == SignalType.FLAGGED_ERROR:
            return (
                "User flagged an error in the response. "
                "Review and consider adding corrective "
                "training examples to Forge."
            )
        return (
            "User rejected the response despite good retrieval. "
            "Consider adding training examples that demonstrate "
            "proper reasoning for this type of query."
        )

    def _generate_reason(self, signal: FeedbackSignal, target: RoutingTarget) -> str:
        """Generate a human-readable reason for the routing.

        Args:
            signal: The feedback signal.
            target: Where this will be routed.

        Returns:
            Explanation of why this routing was chosen.
        """
        reasons: dict[SignalType, str] = {
            SignalType.ACCEPTED: "Positive interaction, no action needed.",
            SignalType.CITATION_CLICKED: ("User clicked a citation, indicating useful retrieval."),
            SignalType.NO_RESULT: (
                "No results found for query, suggesting missing " "documents or indexing gaps."
            ),
            SignalType.REPHRASED: (
                "User rephrased their query, suggesting initial " "retrieval was insufficient."
            ),
            SignalType.FOLLOW_UP: (
                "Follow-up question may indicate incomplete " "coverage in retrieved content."
            ),
            SignalType.FLAGGED_ERROR: ("User explicitly flagged an error in the response."),
        }

        if signal.signal_type == SignalType.REJECTED:
            return self._rejection_reason(signal, target)

        return reasons.get(
            signal.signal_type,
            f"Signal type: {signal.signal_type.value}",
        )

    def _rejection_reason(self, signal: FeedbackSignal, target: RoutingTarget) -> str:
        """Generate reason text for a rejection signal.

        Args:
            signal: The REJECTED feedback signal.
            target: Where this was routed.

        Returns:
            Explanation of why this rejection was routed this way.
        """
        if target == RoutingTarget.QUARRY:
            return (
                "User rejected the response and citation quality "
                "was low, suggesting a retrieval issue."
            )
        return (
            "User rejected the response but citations were "
            "adequate, suggesting a model response quality issue."
        )


# ===================================================================
# PatternAnalyzer
# ===================================================================


class PatternAnalyzer:
    """Identifies patterns from aggregated feedback signals.

    Detects repeated failures, low acceptance areas, and missing
    coverage. All patterns are informational -- they describe issues
    for human review, never auto-fix anything.

    Args:
        min_signals: Minimum number of signals required to form a pattern.
    """

    def __init__(self, min_signals: int = 3) -> None:
        """Initialize with minimum signal threshold.

        Args:
            min_signals: Minimum count to consider a group a pattern.
        """
        self._min_signals = min_signals

    def analyze(self, signals: list[FeedbackSignal]) -> list[FeedbackPattern]:
        """Analyze signals and detect patterns.

        Args:
            signals: List of feedback signals to analyze.

        Returns:
            List of detected patterns, sorted by signal count descending.
        """
        if not signals:
            return []

        patterns: list[FeedbackPattern] = []
        patterns.extend(self._detect_repeated_failures(signals))
        patterns.extend(self._detect_low_acceptance_areas(signals))
        patterns.extend(self._detect_missing_coverage(signals))

        # Deduplicate by pattern_type + affected queries overlap
        seen: set[str] = set()
        unique: list[FeedbackPattern] = []
        for pat in patterns:
            key = f"{pat.pattern_type}:{','.join(sorted(pat.affected_queries))}"
            if key not in seen:
                seen.add(key)
                unique.append(pat)

        return sorted(unique, key=lambda p: p.signal_count, reverse=True)

    def _detect_repeated_failures(self, signals: list[FeedbackSignal]) -> list[FeedbackPattern]:
        """Detect queries that repeatedly fail.

        Groups NO_RESULT and REJECTED signals by query text and
        identifies queries that fail more than the threshold.

        Args:
            signals: All feedback signals.

        Returns:
            List of repeated failure patterns.
        """
        failure_types = {
            SignalType.NO_RESULT,
            SignalType.REJECTED,
            SignalType.FLAGGED_ERROR,
        }
        failures = [s for s in signals if s.signal_type in failure_types]
        return self._group_by_query(failures, "repeated_failures", "Repeated query failures")

    def _detect_low_acceptance_areas(self, signals: list[FeedbackSignal]) -> list[FeedbackPattern]:
        """Detect discipline areas with low acceptance rates.

        Groups REJECTED signals by discipline and identifies areas
        where rejection is common.

        Args:
            signals: All feedback signals.

        Returns:
            List of low acceptance patterns.
        """
        rejections = [s for s in signals if s.signal_type == SignalType.REJECTED]
        return self._group_by_discipline(rejections, "low_acceptance", "Low acceptance rate")

    def _detect_missing_coverage(self, signals: list[FeedbackSignal]) -> list[FeedbackPattern]:
        """Detect areas with no retrieval results.

        Groups NO_RESULT signals to find topic areas that are not
        covered by the current document set.

        Args:
            signals: All feedback signals.

        Returns:
            List of missing coverage patterns.
        """
        no_results = [s for s in signals if s.signal_type == SignalType.NO_RESULT]
        return self._group_by_query(no_results, "missing_coverage", "Missing document coverage")

    def _group_by_query(
        self,
        signals: list[FeedbackSignal],
        pattern_type: str,
        description_prefix: str,
    ) -> list[FeedbackPattern]:
        """Group signals by query text and create patterns.

        Args:
            signals: Filtered signals to group.
            pattern_type: Type label for resulting patterns.
            description_prefix: Start of the description string.

        Returns:
            Patterns for query groups exceeding the threshold.
        """
        query_groups: dict[str, list[FeedbackSignal]] = {}
        for sig in signals:
            query_groups.setdefault(sig.query, []).append(sig)

        patterns: list[FeedbackPattern] = []
        for query, group in query_groups.items():
            if len(group) >= self._min_signals:
                patterns.append(
                    self._build_pattern(group, pattern_type, description_prefix, [query])
                )
        return patterns

    def _group_by_discipline(
        self,
        signals: list[FeedbackSignal],
        pattern_type: str,
        description_prefix: str,
    ) -> list[FeedbackPattern]:
        """Group signals by discipline and create patterns.

        Args:
            signals: Filtered signals to group.
            pattern_type: Type label for resulting patterns.
            description_prefix: Start of the description string.

        Returns:
            Patterns for discipline groups exceeding the threshold.
        """
        disc_groups: dict[str, list[FeedbackSignal]] = {}
        for sig in signals:
            key = sig.discipline_id or "unknown"
            disc_groups.setdefault(key, []).append(sig)

        patterns: list[FeedbackPattern] = []
        for _disc_id, group in disc_groups.items():
            if len(group) >= self._min_signals:
                queries = list({s.query for s in group})
                patterns.append(
                    self._build_pattern(group, pattern_type, description_prefix, queries)
                )
        return patterns

    def _build_pattern(
        self,
        signals: list[FeedbackSignal],
        pattern_type: str,
        description_prefix: str,
        affected_queries: list[str],
    ) -> FeedbackPattern:
        """Build a FeedbackPattern from a group of signals.

        Args:
            signals: Signals forming this pattern.
            pattern_type: Category of the pattern.
            description_prefix: Human-readable prefix.
            affected_queries: Queries involved in this pattern.

        Returns:
            A constructed FeedbackPattern.
        """
        timestamps = [s.timestamp for s in signals]
        first_seen = min(timestamps)
        last_seen = max(timestamps)

        # Route based on the most common signal type in this group
        type_counts = Counter(s.signal_type for s in signals)
        dominant_type = type_counts.most_common(1)[0][0]
        router = SignalRouter()
        sample_decision = router.route(signals[0])

        description = (
            f"{description_prefix}: {len(signals)} signals "
            f"involving {len(affected_queries)} "
            f"{'query' if len(affected_queries) == 1 else 'queries'}"
        )

        target = sample_decision.target
        if target == RoutingTarget.NONE:
            target = self._infer_target(dominant_type)

        return FeedbackPattern(
            pattern_id=FeedbackPattern.generate_id(),
            pattern_type=pattern_type,
            description=description,
            affected_queries=affected_queries,
            signal_count=len(signals),
            first_seen=first_seen,
            last_seen=last_seen,
            routing=RoutingDecision(
                target=target,
                priority=sample_decision.priority,
                reason=description,
                suggested_action=sample_decision.suggested_action,
                signal_id=sample_decision.signal_id,
            ),
        )

    def _infer_target(self, dominant_type: SignalType) -> RoutingTarget:
        """Infer routing target from dominant signal type.

        Args:
            dominant_type: The most common signal type in a group.

        Returns:
            Best-guess routing target.
        """
        quarry_types = {
            SignalType.NO_RESULT,
            SignalType.REPHRASED,
            SignalType.FOLLOW_UP,
        }
        forge_types = {
            SignalType.FLAGGED_ERROR,
            SignalType.REJECTED,
        }
        if dominant_type in quarry_types:
            return RoutingTarget.QUARRY
        if dominant_type in forge_types:
            return RoutingTarget.FORGE
        return RoutingTarget.NONE


# ===================================================================
# FeedbackManager
# ===================================================================


class FeedbackManager:
    """Main orchestrator for feedback capture and routing.

    Coordinates signal capture, storage, routing, pattern analysis,
    and dashboard generation. All outputs are SUGGESTIONS for human
    discipline owners. No automated actions are ever taken.

    Attributes:
        store: The signal storage backend.
        router: The signal routing engine.
        analyzer: The pattern detection engine.
    """

    def __init__(
        self,
        store: FeedbackStore | None = None,
        router: SignalRouter | None = None,
        analyzer: PatternAnalyzer | None = None,
    ) -> None:
        """Initialize with optional custom components.

        Args:
            store: Custom FeedbackStore (defaults to new instance).
            router: Custom SignalRouter (defaults to new instance).
            analyzer: Custom PatternAnalyzer (defaults to new instance).
        """
        self.store = store or FeedbackStore()
        self.router = router or SignalRouter()
        self.analyzer = analyzer or PatternAnalyzer()

    def capture(
        self,
        signal_type: SignalType,
        conversation_id: str,
        query: str,
        response: str | None = None,
        user_comment: str | None = None,
        discipline_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackSignal:
        """Capture a feedback signal from a user interaction.

        Args:
            signal_type: Category of the interaction.
            conversation_id: Which conversation this belongs to.
            query: The user's query text.
            response: The system's response (optional).
            user_comment: Free-text user feedback (optional).
            discipline_id: Related discipline (optional).
            metadata: Extra context dict (optional).

        Returns:
            The recorded FeedbackSignal.
        """
        signal = FeedbackSignal(
            signal_id=FeedbackSignal.generate_id(),
            signal_type=signal_type,
            conversation_id=conversation_id,
            query=query,
            response=response,
            user_comment=user_comment,
            discipline_id=discipline_id,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.store.record(signal)
        return signal

    def get_routing(self, signal_id: str) -> RoutingDecision:
        """Get the routing decision for a specific signal.

        Args:
            signal_id: ID of the signal to route.

        Returns:
            The routing decision for the signal.

        Raises:
            FeedbackError: If the signal is not found.
        """
        signal = self._find_signal(signal_id)
        return self.router.route(signal)

    def get_dashboard(self, discipline_id: str, days: int = 30) -> DashboardSummary:
        """Generate a dashboard summary for a discipline.

        Args:
            discipline_id: Which discipline to summarize.
            days: Number of days to look back.

        Returns:
            DashboardSummary with rates, counts, and suggestions.
        """
        now = datetime.now()
        period_start = now - timedelta(days=days)
        signals = self.store.get_signals(discipline_id=discipline_id)
        recent = [s for s in signals if s.timestamp >= period_start]

        return self._build_dashboard(discipline_id, recent, period_start, now)

    def _build_dashboard(
        self,
        discipline_id: str,
        signals: list[FeedbackSignal],
        period_start: datetime,
        period_end: datetime,
    ) -> DashboardSummary:
        """Build a DashboardSummary from filtered signals.

        Args:
            discipline_id: The discipline being summarized.
            signals: Filtered signals for the period.
            period_start: Start of the reporting period.
            period_end: End of the reporting period.

        Returns:
            Constructed DashboardSummary.
        """
        total = len(signals)
        accepted = sum(1 for s in signals if s.signal_type == SignalType.ACCEPTED)
        rejected = sum(1 for s in signals if s.signal_type == SignalType.REJECTED)
        flagged = sum(1 for s in signals if s.signal_type == SignalType.FLAGGED_ERROR)

        acceptance_rate = accepted / total if total > 0 else 0.0
        rejection_rate = rejected / total if total > 0 else 0.0

        patterns = self.analyzer.analyze(signals)
        decisions = self.router.route_batch(signals)

        quarry_actions = [d for d in decisions if d.target == RoutingTarget.QUARRY]
        forge_actions = [d for d in decisions if d.target == RoutingTarget.FORGE]

        return DashboardSummary(
            discipline_id=discipline_id,
            total_queries=total,
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
            flagged_errors=flagged,
            top_issues=patterns,
            quarry_actions=quarry_actions,
            forge_actions=forge_actions,
            period_start=period_start,
            period_end=period_end,
        )

    def get_patterns(self, discipline_id: str | None = None) -> list[FeedbackPattern]:
        """Get detected patterns, optionally filtered by discipline.

        Args:
            discipline_id: Filter to a specific discipline (optional).

        Returns:
            List of detected FeedbackPatterns.
        """
        signals = self.store.get_signals(discipline_id=discipline_id)
        return self.analyzer.analyze(signals)

    def export_signals(self, path: Path) -> int:
        """Export all signals to a JSONL file.

        Exports signal data (NOT training data format). Each line
        is a JSON object representing one FeedbackSignal.

        Args:
            path: File path to write the JSONL export.

        Returns:
            Number of signals exported.
        """
        signals = self.store.get_signals()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            for signal in signals:
                fh.write(json.dumps(signal.to_dict()) + "\n")

        return len(signals)

    def _find_signal(self, signal_id: str) -> FeedbackSignal:
        """Find a signal by ID.

        Args:
            signal_id: The signal ID to look up.

        Returns:
            The matching FeedbackSignal.

        Raises:
            FeedbackError: If no signal with this ID exists.
        """
        for signal in self.store.get_signals():
            if signal.signal_id == signal_id:
                return signal
        raise FeedbackError(f"Signal '{signal_id}' not found")
