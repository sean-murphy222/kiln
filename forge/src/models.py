"""Forge data models for curriculum building.

Defines domain models for disciplines, competencies, examples,
contributors, and curriculum versioning. All models use dataclasses
with serialization support and UUID-based ID generation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ContributorRole(str, Enum):
    """Role a contributor plays within a discipline."""

    CONTRIBUTOR = "contributor"
    LEAD = "lead"
    ADMIN = "admin"


class DisciplineStatus(str, Enum):
    """Lifecycle status of a discipline."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ReviewStatus(str, Enum):
    """Review status of a training example."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class CurriculumStatus(str, Enum):
    """Status of a curriculum version snapshot."""

    DRAFT = "draft"
    PUBLISHED = "published"


@dataclass
class Contributor:
    """A person who contributes training examples to a discipline.

    Attributes:
        id: Unique identifier (prefixed with 'contrib_').
        name: Display name.
        email: Contact email (optional).
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    id: str
    name: str
    email: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique contributor ID."""
        return f"contrib_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contributor:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            email=data.get("email", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Discipline:
    """A domain of expertise for training (e.g., military maintenance).

    Attributes:
        id: Unique identifier (prefixed with 'disc_').
        name: Human-readable discipline name.
        description: What this discipline covers.
        status: Current lifecycle status.
        created_by: Contributor ID of the creator.
        vocabulary: Domain-specific terms and definitions.
        document_types: Types of documents used in this discipline.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    id: str
    name: str
    description: str
    status: DisciplineStatus = DisciplineStatus.DRAFT
    created_by: str = ""
    vocabulary: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique discipline ID."""
        return f"disc_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_by": self.created_by,
            "vocabulary": self.vocabulary,
            "document_types": self.document_types,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Discipline:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=DisciplineStatus(data["status"]),
            created_by=data.get("created_by", ""),
            vocabulary=data.get("vocabulary", []),
            document_types=data.get("document_types", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Competency:
    """A specific skill or knowledge area within a discipline.

    Attributes:
        id: Unique identifier (prefixed with 'comp_').
        name: Human-readable competency name.
        description: What this competency covers.
        discipline_id: Parent discipline ID.
        parent_id: Parent competency ID (for hierarchical competencies).
        coverage_target: Number of examples needed for full coverage.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    id: str
    name: str
    description: str
    discipline_id: str
    parent_id: str | None = None
    coverage_target: int = 25
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique competency ID."""
        return f"comp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "discipline_id": self.discipline_id,
            "parent_id": self.parent_id,
            "coverage_target": self.coverage_target,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Competency:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            discipline_id=data["discipline_id"],
            parent_id=data.get("parent_id"),
            coverage_target=data.get("coverage_target", 25),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Example:
    """A single training example for a competency.

    Attributes:
        id: Unique identifier (prefixed with 'ex_').
        question: The question/prompt (instruction field in Alpaca format).
        ideal_answer: The ideal response (output field in Alpaca format).
        competency_id: Parent competency ID.
        contributor_id: Who created this example.
        discipline_id: Parent discipline ID (denormalized for queries).
        variants: Alternative phrasings of the question.
        context: Additional context for the example.
        review_status: Current review status.
        reviewed_by: Contributor ID of the reviewer.
        reviewed_at: When the review was completed.
        is_test_set: Whether this is held out for evaluation.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    id: str
    question: str
    ideal_answer: str
    competency_id: str
    contributor_id: str
    discipline_id: str
    variants: list[str] = field(default_factory=list)
    context: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    is_test_set: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique example ID."""
        return f"ex_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "ideal_answer": self.ideal_answer,
            "competency_id": self.competency_id,
            "contributor_id": self.contributor_id,
            "discipline_id": self.discipline_id,
            "variants": self.variants,
            "context": self.context,
            "review_status": self.review_status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "is_test_set": self.is_test_set,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example:
        """Deserialize from dictionary."""
        reviewed_at = data.get("reviewed_at")
        if reviewed_at and isinstance(reviewed_at, str):
            reviewed_at = datetime.fromisoformat(reviewed_at)
        return cls(
            id=data["id"],
            question=data["question"],
            ideal_answer=data["ideal_answer"],
            competency_id=data["competency_id"],
            contributor_id=data["contributor_id"],
            discipline_id=data["discipline_id"],
            variants=data.get("variants", []),
            context=data.get("context", ""),
            review_status=ReviewStatus(data.get("review_status", "pending")),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=reviewed_at,
            is_test_set=data.get("is_test_set", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def to_training_record(self) -> dict[str, Any]:
        """Convert to Alpaca-format training record for JSONL export.

        Returns:
            Dict with instruction, input (always empty), output, and metadata.
        """
        return {
            "instruction": self.question,
            "input": "",
            "output": self.ideal_answer,
            "metadata": {
                "example_id": self.id,
                "discipline_id": self.discipline_id,
                "competency_id": self.competency_id,
                "contributor_id": self.contributor_id,
                "review_status": self.review_status.value,
                "created_at": self.created_at.isoformat(),
            },
        }


@dataclass
class DisciplineContributor:
    """Association between a discipline and a contributor.

    Attributes:
        discipline_id: The discipline ID.
        contributor_id: The contributor ID.
        role: Role within this discipline.
        competency_area_ids: Competencies this contributor focuses on.
        joined_at: When they joined this discipline.
    """

    discipline_id: str
    contributor_id: str
    role: ContributorRole = ContributorRole.CONTRIBUTOR
    competency_area_ids: list[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "discipline_id": self.discipline_id,
            "contributor_id": self.contributor_id,
            "role": self.role.value,
            "competency_area_ids": self.competency_area_ids,
            "joined_at": self.joined_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisciplineContributor:
        """Deserialize from dictionary."""
        return cls(
            discipline_id=data["discipline_id"],
            contributor_id=data["contributor_id"],
            role=ContributorRole(data["role"]),
            competency_area_ids=data.get("competency_area_ids", []),
            joined_at=datetime.fromisoformat(data["joined_at"]),
        )


@dataclass
class CurriculumVersion:
    """Immutable snapshot of a discipline's training examples.

    Attributes:
        id: Unique identifier (prefixed with 'curv_').
        discipline_id: The discipline this version belongs to.
        version_number: Sequential version number.
        created_by: Contributor who created this snapshot.
        example_count: Number of examples in the snapshot.
        snapshot_json: JSON-serialized list of training examples.
        status: Publication status.
        created_at: Creation timestamp.
    """

    id: str
    discipline_id: str
    version_number: int
    created_by: str
    example_count: int
    snapshot_json: str = ""
    status: CurriculumStatus = CurriculumStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique curriculum version ID."""
        return f"curv_{uuid.uuid4().hex[:12]}"

    def get_examples(self) -> list[Example]:
        """Deserialize the snapshot into Example objects.

        Returns:
            List of Example objects from the snapshot.
        """
        if not self.snapshot_json:
            return []
        data = json.loads(self.snapshot_json)
        return [Example.from_dict(d) for d in data]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "discipline_id": self.discipline_id,
            "version_number": self.version_number,
            "created_by": self.created_by,
            "example_count": self.example_count,
            "snapshot_json": self.snapshot_json,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumVersion:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            discipline_id=data["discipline_id"],
            version_number=data["version_number"],
            created_by=data["created_by"],
            example_count=data["example_count"],
            snapshot_json=data.get("snapshot_json", ""),
            status=CurriculumStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
