"""SQLite-backed storage for Forge data models.

Provides CRUD operations, curriculum versioning, coverage reporting,
and JSONL export for Foundry consumption.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from forge.src.models import (
    Competency,
    Contributor,
    ContributorRole,
    CurriculumStatus,
    CurriculumVersion,
    Discipline,
    DisciplineContributor,
    DisciplineStatus,
    Example,
    ReviewStatus,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS contributors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disciplines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_by TEXT NOT NULL,
    vocabulary_json TEXT DEFAULT '[]',
    document_types_json TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (created_by) REFERENCES contributors(id)
);

CREATE TABLE IF NOT EXISTS competencies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    discipline_id TEXT NOT NULL,
    parent_id TEXT,
    coverage_target INTEGER DEFAULT 25,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (discipline_id) REFERENCES disciplines(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES competencies(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS examples (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    ideal_answer TEXT NOT NULL,
    competency_id TEXT NOT NULL,
    contributor_id TEXT NOT NULL,
    discipline_id TEXT NOT NULL,
    variants_json TEXT DEFAULT '[]',
    context TEXT DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewed_by TEXT,
    reviewed_at TEXT,
    is_test_set INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (competency_id) REFERENCES competencies(id) ON DELETE CASCADE,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (discipline_id) REFERENCES disciplines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS discipline_contributors (
    discipline_id TEXT NOT NULL,
    contributor_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'contributor',
    competency_area_ids_json TEXT DEFAULT '[]',
    joined_at TEXT NOT NULL,
    PRIMARY KEY (discipline_id, contributor_id),
    FOREIGN KEY (discipline_id) REFERENCES disciplines(id) ON DELETE CASCADE,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS curriculum_versions (
    id TEXT PRIMARY KEY,
    discipline_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    created_by TEXT NOT NULL,
    example_count INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL,
    FOREIGN KEY (discipline_id) REFERENCES disciplines(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES contributors(id)
);

CREATE INDEX IF NOT EXISTS idx_examples_discipline
    ON examples(discipline_id);
CREATE INDEX IF NOT EXISTS idx_examples_competency
    ON examples(competency_id);
CREATE INDEX IF NOT EXISTS idx_examples_discipline_test
    ON examples(discipline_id, is_test_set);
CREATE INDEX IF NOT EXISTS idx_competencies_discipline
    ON competencies(discipline_id);
CREATE INDEX IF NOT EXISTS idx_curriculum_versions_discipline
    ON curriculum_versions(discipline_id);
"""


class ForgeStorageError(Exception):
    """Raised for storage-level errors (duplicates, not found, etc.)."""


class ForgeStorage:
    """SQLite-backed storage for Forge domain models.

    Supports CRUD for contributors, disciplines, competencies, examples,
    discipline-contributor associations, and curriculum versioning.

    Args:
        db_path: Path to SQLite database file, or ':memory:' for in-memory.

    Example::

        with ForgeStorage("forge.db") as store:
            store.initialize_schema()
            store.create_contributor(contributor)
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row

    def __enter__(self) -> ForgeStorage:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the database connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def initialize_schema(self) -> None:
        """Create all tables and indexes if they don't exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ---------------------------------------------------------------
    # Contributors
    # ---------------------------------------------------------------

    def create_contributor(self, contributor: Contributor) -> Contributor:
        """Insert a new contributor.

        Args:
            contributor: Contributor to insert.

        Returns:
            The inserted contributor.

        Raises:
            ForgeStorageError: If a contributor with the same ID exists.
        """
        try:
            self._conn.execute(
                "INSERT INTO contributors (id, name, email, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    contributor.id,
                    contributor.name,
                    contributor.email,
                    contributor.created_at.isoformat(),
                    contributor.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ForgeStorageError(f"Contributor already exists: {contributor.id}") from exc
        return contributor

    def get_contributor(self, contributor_id: str) -> Contributor | None:
        """Fetch a contributor by ID.

        Args:
            contributor_id: The contributor's unique ID.

        Returns:
            Contributor or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM contributors WHERE id = ?", (contributor_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_contributor(row)

    def update_contributor(self, contributor: Contributor) -> Contributor:
        """Update an existing contributor.

        Args:
            contributor: Contributor with updated fields.

        Returns:
            The updated contributor with refreshed updated_at.

        Raises:
            ForgeStorageError: If contributor does not exist.
        """
        contributor.updated_at = datetime.now()
        cursor = self._conn.execute(
            "UPDATE contributors SET name = ?, email = ?, updated_at = ? WHERE id = ?",
            (
                contributor.name,
                contributor.email,
                contributor.updated_at.isoformat(),
                contributor.id,
            ),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError(f"Contributor not found: {contributor.id}")
        self._conn.commit()
        return contributor

    def delete_contributor(self, contributor_id: str) -> bool:
        """Delete a contributor by ID.

        Args:
            contributor_id: ID of the contributor to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute("DELETE FROM contributors WHERE id = ?", (contributor_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------------------------------------------------------------
    # Disciplines
    # ---------------------------------------------------------------

    def create_discipline(self, discipline: Discipline) -> Discipline:
        """Insert a new discipline.

        Args:
            discipline: Discipline to insert.

        Returns:
            The inserted discipline.

        Raises:
            ForgeStorageError: If a discipline with the same ID exists.
        """
        try:
            self._conn.execute(
                "INSERT INTO disciplines "
                "(id, name, description, status, created_by, "
                "vocabulary_json, document_types_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    discipline.id,
                    discipline.name,
                    discipline.description,
                    discipline.status.value,
                    discipline.created_by,
                    json.dumps(discipline.vocabulary),
                    json.dumps(discipline.document_types),
                    discipline.created_at.isoformat(),
                    discipline.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ForgeStorageError(f"Discipline creation failed: {discipline.id}") from exc
        return discipline

    def get_discipline(self, discipline_id: str) -> Discipline | None:
        """Fetch a discipline by ID.

        Args:
            discipline_id: The discipline's unique ID.

        Returns:
            Discipline or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM disciplines WHERE id = ?", (discipline_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_discipline(row)

    def get_all_disciplines(self, status: DisciplineStatus | None = None) -> list[Discipline]:
        """Fetch all disciplines, optionally filtered by status.

        Args:
            status: If provided, only return disciplines with this status.

        Returns:
            List of matching disciplines.
        """
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM disciplines WHERE status = ?", (status.value,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM disciplines").fetchall()
        return [self._row_to_discipline(r) for r in rows]

    def update_discipline(self, discipline: Discipline) -> Discipline:
        """Update an existing discipline.

        Args:
            discipline: Discipline with updated fields.

        Returns:
            The updated discipline with refreshed updated_at.

        Raises:
            ForgeStorageError: If discipline does not exist.
        """
        discipline.updated_at = datetime.now()
        cursor = self._conn.execute(
            "UPDATE disciplines SET name = ?, description = ?, status = ?, "
            "vocabulary_json = ?, document_types_json = ?, updated_at = ? "
            "WHERE id = ?",
            (
                discipline.name,
                discipline.description,
                discipline.status.value,
                json.dumps(discipline.vocabulary),
                json.dumps(discipline.document_types),
                discipline.updated_at.isoformat(),
                discipline.id,
            ),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError(f"Discipline not found: {discipline.id}")
        self._conn.commit()
        return discipline

    def delete_discipline(self, discipline_id: str) -> bool:
        """Delete a discipline by ID (cascades to competencies/examples).

        Args:
            discipline_id: ID of the discipline to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute("DELETE FROM disciplines WHERE id = ?", (discipline_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------------------------------------------------------------
    # Competencies
    # ---------------------------------------------------------------

    def create_competency(self, competency: Competency) -> Competency:
        """Insert a new competency.

        Args:
            competency: Competency to insert.

        Returns:
            The inserted competency.

        Raises:
            ForgeStorageError: If creation fails (FK violation, duplicate).
        """
        try:
            self._conn.execute(
                "INSERT INTO competencies "
                "(id, name, description, discipline_id, parent_id, "
                "coverage_target, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    competency.id,
                    competency.name,
                    competency.description,
                    competency.discipline_id,
                    competency.parent_id,
                    competency.coverage_target,
                    competency.created_at.isoformat(),
                    competency.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ForgeStorageError(f"Competency creation failed: {competency.id}") from exc
        return competency

    def get_competency(self, competency_id: str) -> Competency | None:
        """Fetch a competency by ID.

        Args:
            competency_id: The competency's unique ID.

        Returns:
            Competency or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM competencies WHERE id = ?", (competency_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_competency(row)

    def get_competencies_for_discipline(self, discipline_id: str) -> list[Competency]:
        """Fetch all competencies for a discipline.

        Args:
            discipline_id: Parent discipline ID.

        Returns:
            List of competencies for the discipline.
        """
        rows = self._conn.execute(
            "SELECT * FROM competencies WHERE discipline_id = ?",
            (discipline_id,),
        ).fetchall()
        return [self._row_to_competency(r) for r in rows]

    def update_competency(self, competency: Competency) -> Competency:
        """Update an existing competency.

        Args:
            competency: Competency with updated fields.

        Returns:
            The updated competency with refreshed updated_at.

        Raises:
            ForgeStorageError: If competency does not exist.
        """
        competency.updated_at = datetime.now()
        cursor = self._conn.execute(
            "UPDATE competencies SET name = ?, description = ?, "
            "parent_id = ?, coverage_target = ?, updated_at = ? "
            "WHERE id = ?",
            (
                competency.name,
                competency.description,
                competency.parent_id,
                competency.coverage_target,
                competency.updated_at.isoformat(),
                competency.id,
            ),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError(f"Competency not found: {competency.id}")
        self._conn.commit()
        return competency

    def delete_competency(self, competency_id: str) -> bool:
        """Delete a competency by ID (cascades to examples).

        Args:
            competency_id: ID of the competency to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute("DELETE FROM competencies WHERE id = ?", (competency_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------------------------------------------------------------
    # Examples
    # ---------------------------------------------------------------

    def create_example(self, example: Example) -> Example:
        """Insert a new example.

        Args:
            example: Example to insert.

        Returns:
            The inserted example.

        Raises:
            ForgeStorageError: If creation fails.
        """
        try:
            self._conn.execute(
                "INSERT INTO examples "
                "(id, question, ideal_answer, competency_id, contributor_id, "
                "discipline_id, variants_json, context, review_status, "
                "reviewed_by, reviewed_at, is_test_set, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    example.id,
                    example.question,
                    example.ideal_answer,
                    example.competency_id,
                    example.contributor_id,
                    example.discipline_id,
                    json.dumps(example.variants),
                    example.context,
                    example.review_status.value,
                    example.reviewed_by,
                    example.reviewed_at.isoformat() if example.reviewed_at else None,
                    1 if example.is_test_set else 0,
                    example.created_at.isoformat(),
                    example.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ForgeStorageError(f"Example creation failed: {example.id}") from exc
        return example

    def get_example(self, example_id: str) -> Example | None:
        """Fetch an example by ID.

        Args:
            example_id: The example's unique ID.

        Returns:
            Example or None if not found.
        """
        row = self._conn.execute("SELECT * FROM examples WHERE id = ?", (example_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_example(row)

    def get_examples_for_competency(
        self, competency_id: str, include_test_set: bool = True
    ) -> list[Example]:
        """Fetch examples for a competency.

        Args:
            competency_id: Parent competency ID.
            include_test_set: If False, excludes test-set examples.

        Returns:
            List of matching examples.
        """
        if include_test_set:
            rows = self._conn.execute(
                "SELECT * FROM examples WHERE competency_id = ?",
                (competency_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM examples WHERE competency_id = ? AND is_test_set = 0",
                (competency_id,),
            ).fetchall()
        return [self._row_to_example(r) for r in rows]

    def get_training_examples(self, discipline_id: str) -> list[Example]:
        """Fetch all training (non-test-set) examples for a discipline.

        Args:
            discipline_id: The discipline ID.

        Returns:
            List of training examples.
        """
        rows = self._conn.execute(
            "SELECT * FROM examples WHERE discipline_id = ? AND is_test_set = 0",
            (discipline_id,),
        ).fetchall()
        return [self._row_to_example(r) for r in rows]

    def get_test_set_examples(self, discipline_id: str) -> list[Example]:
        """Fetch all test-set examples for a discipline.

        Args:
            discipline_id: The discipline ID.

        Returns:
            List of test-set examples.
        """
        rows = self._conn.execute(
            "SELECT * FROM examples WHERE discipline_id = ? AND is_test_set = 1",
            (discipline_id,),
        ).fetchall()
        return [self._row_to_example(r) for r in rows]

    def update_example(self, example: Example) -> Example:
        """Update an existing example.

        Args:
            example: Example with updated fields.

        Returns:
            The updated example with refreshed updated_at.

        Raises:
            ForgeStorageError: If example does not exist.
        """
        example.updated_at = datetime.now()
        cursor = self._conn.execute(
            "UPDATE examples SET question = ?, ideal_answer = ?, "
            "variants_json = ?, context = ?, review_status = ?, "
            "reviewed_by = ?, reviewed_at = ?, is_test_set = ?, updated_at = ? "
            "WHERE id = ?",
            (
                example.question,
                example.ideal_answer,
                json.dumps(example.variants),
                example.context,
                example.review_status.value,
                example.reviewed_by,
                example.reviewed_at.isoformat() if example.reviewed_at else None,
                1 if example.is_test_set else 0,
                example.updated_at.isoformat(),
                example.id,
            ),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError(f"Example not found: {example.id}")
        self._conn.commit()
        return example

    def delete_example(self, example_id: str) -> bool:
        """Delete an example by ID.

        Args:
            example_id: ID of the example to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute("DELETE FROM examples WHERE id = ?", (example_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------------------------------------------------------------
    # Discipline Contributors
    # ---------------------------------------------------------------

    def add_contributor_to_discipline(self, dc: DisciplineContributor) -> DisciplineContributor:
        """Add a contributor to a discipline.

        Args:
            dc: DisciplineContributor association.

        Returns:
            The inserted association.

        Raises:
            ForgeStorageError: If the pair already exists.
        """
        try:
            self._conn.execute(
                "INSERT INTO discipline_contributors "
                "(discipline_id, contributor_id, role, competency_area_ids_json, joined_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    dc.discipline_id,
                    dc.contributor_id,
                    dc.role.value,
                    json.dumps(dc.competency_area_ids),
                    dc.joined_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ForgeStorageError("Contributor already in discipline") from exc
        return dc

    def get_discipline_contributors(self, discipline_id: str) -> list[DisciplineContributor]:
        """Get all contributors for a discipline.

        Args:
            discipline_id: The discipline ID.

        Returns:
            List of DisciplineContributor associations.
        """
        rows = self._conn.execute(
            "SELECT * FROM discipline_contributors WHERE discipline_id = ?",
            (discipline_id,),
        ).fetchall()
        return [self._row_to_discipline_contributor(r) for r in rows]

    def update_contributor_in_discipline(self, dc: DisciplineContributor) -> DisciplineContributor:
        """Update a contributor's role or competency areas in a discipline.

        Args:
            dc: Updated association.

        Returns:
            The updated association.

        Raises:
            ForgeStorageError: If the association does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE discipline_contributors SET role = ?, competency_area_ids_json = ? "
            "WHERE discipline_id = ? AND contributor_id = ?",
            (
                dc.role.value,
                json.dumps(dc.competency_area_ids),
                dc.discipline_id,
                dc.contributor_id,
            ),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError("Contributor not in discipline")
        self._conn.commit()
        return dc

    def remove_contributor_from_discipline(self, discipline_id: str, contributor_id: str) -> bool:
        """Remove a contributor from a discipline.

        Args:
            discipline_id: The discipline ID.
            contributor_id: The contributor ID.

        Returns:
            True if removed, False if not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM discipline_contributors " "WHERE discipline_id = ? AND contributor_id = ?",
            (discipline_id, contributor_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------------------------------------------------------------
    # Curriculum Versioning
    # ---------------------------------------------------------------

    def create_curriculum_version(self, discipline_id: str, created_by: str) -> CurriculumVersion:
        """Create a new curriculum version snapshot.

        Snapshots all training (non-test-set) examples for the discipline
        and stores them as a JSON blob.

        Args:
            discipline_id: The discipline to snapshot.
            created_by: Contributor ID creating this version.

        Returns:
            The new CurriculumVersion with snapshot.
        """
        # Get next version number
        row = self._conn.execute(
            "SELECT MAX(version_number) FROM curriculum_versions " "WHERE discipline_id = ?",
            (discipline_id,),
        ).fetchone()
        next_version = (row[0] or 0) + 1

        # Snapshot training examples
        examples = self.get_training_examples(discipline_id)
        snapshot = json.dumps([e.to_dict() for e in examples])

        version = CurriculumVersion(
            id=CurriculumVersion.generate_id(),
            discipline_id=discipline_id,
            version_number=next_version,
            created_by=created_by,
            example_count=len(examples),
            snapshot_json=snapshot,
            status=CurriculumStatus.DRAFT,
        )

        self._conn.execute(
            "INSERT INTO curriculum_versions "
            "(id, discipline_id, version_number, created_by, "
            "example_count, snapshot_json, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version.id,
                version.discipline_id,
                version.version_number,
                version.created_by,
                version.example_count,
                version.snapshot_json,
                version.status.value,
                version.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        return version

    def get_latest_curriculum_version(self, discipline_id: str) -> CurriculumVersion | None:
        """Get the latest curriculum version for a discipline.

        Args:
            discipline_id: The discipline ID.

        Returns:
            Latest CurriculumVersion or None if no versions exist.
        """
        row = self._conn.execute(
            "SELECT * FROM curriculum_versions "
            "WHERE discipline_id = ? ORDER BY version_number DESC LIMIT 1",
            (discipline_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_curriculum_version(row)

    def publish_curriculum_version(self, version_id: str) -> CurriculumVersion:
        """Publish a curriculum version.

        Args:
            version_id: ID of the version to publish.

        Returns:
            The published CurriculumVersion.

        Raises:
            ForgeStorageError: If version does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE curriculum_versions SET status = ? WHERE id = ?",
            (CurriculumStatus.PUBLISHED.value, version_id),
        )
        if cursor.rowcount == 0:
            raise ForgeStorageError(f"Curriculum version not found: {version_id}")
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM curriculum_versions WHERE id = ?", (version_id,)
        ).fetchone()
        return self._row_to_curriculum_version(row)

    def get_curriculum_history(self, discipline_id: str) -> list[CurriculumVersion]:
        """Get all curriculum versions for a discipline, newest first.

        Args:
            discipline_id: The discipline ID.

        Returns:
            List of CurriculumVersion ordered by version_number descending.
        """
        rows = self._conn.execute(
            "SELECT * FROM curriculum_versions "
            "WHERE discipline_id = ? ORDER BY version_number DESC",
            (discipline_id,),
        ).fetchall()
        return [self._row_to_curriculum_version(r) for r in rows]

    # ---------------------------------------------------------------
    # Coverage Report
    # ---------------------------------------------------------------

    def get_coverage_report(self, discipline_id: str) -> dict[str, Any]:
        """Generate a competency coverage report for a discipline.

        Reports how many training examples exist per competency vs.
        target, identifies gaps, and computes overall completeness.

        Args:
            discipline_id: The discipline to report on.

        Returns:
            Dict with discipline_id, total_examples, total_test_examples,
            competency_coverage, gaps, and coverage_complete.
        """
        competencies = self.get_competencies_for_discipline(discipline_id)

        # Count training and test examples per competency
        training_rows = self._conn.execute(
            "SELECT competency_id, COUNT(*) as cnt "
            "FROM examples WHERE discipline_id = ? AND is_test_set = 0 "
            "GROUP BY competency_id",
            (discipline_id,),
        ).fetchall()
        training_counts: dict[str, int] = {r["competency_id"]: r["cnt"] for r in training_rows}

        total_test = self._conn.execute(
            "SELECT COUNT(*) FROM examples WHERE discipline_id = ? AND is_test_set = 1",
            (discipline_id,),
        ).fetchone()[0]

        coverage = []
        gaps = []
        total_examples = 0

        for comp in competencies:
            count = training_counts.get(comp.id, 0)
            total_examples += count
            entry = {
                "competency_id": comp.id,
                "competency_name": comp.name,
                "example_count": count,
                "coverage_target": comp.coverage_target,
                "met": count >= comp.coverage_target,
            }
            coverage.append(entry)
            if not entry["met"]:
                gaps.append(entry)

        return {
            "discipline_id": discipline_id,
            "total_examples": total_examples,
            "total_test_examples": total_test,
            "competency_coverage": coverage,
            "gaps": gaps,
            "coverage_complete": len(gaps) == 0,
        }

    # ---------------------------------------------------------------
    # JSONL Export
    # ---------------------------------------------------------------

    def export_to_jsonl(
        self,
        discipline_id: str,
        output_path: str | Path,
        include_test_set: bool = False,
    ) -> Path:
        """Export training examples to JSONL in Alpaca format.

        Args:
            discipline_id: The discipline to export.
            output_path: Destination file path.
            include_test_set: If True, include test-set examples.

        Returns:
            Path to the written file.
        """
        path = Path(output_path)
        if include_test_set:
            rows = self._conn.execute(
                "SELECT * FROM examples WHERE discipline_id = ?",
                (discipline_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM examples WHERE discipline_id = ? AND is_test_set = 0",
                (discipline_id,),
            ).fetchall()

        examples = [self._row_to_example(r) for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_training_record()) + "\n")
        return path

    def export_test_set_jsonl(self, discipline_id: str, output_path: str | Path) -> Path:
        """Export test-set examples to JSONL in Alpaca format.

        Args:
            discipline_id: The discipline to export.
            output_path: Destination file path.

        Returns:
            Path to the written file.
        """
        path = Path(output_path)
        examples = self.get_test_set_examples(discipline_id)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_training_record()) + "\n")
        return path

    # ---------------------------------------------------------------
    # Row-to-model helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _row_to_contributor(row: sqlite3.Row) -> Contributor:
        """Convert a database row to a Contributor."""
        return Contributor(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_discipline(row: sqlite3.Row) -> Discipline:
        """Convert a database row to a Discipline."""
        return Discipline(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=DisciplineStatus(row["status"]),
            created_by=row["created_by"],
            vocabulary=json.loads(row["vocabulary_json"]),
            document_types=json.loads(row["document_types_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_competency(row: sqlite3.Row) -> Competency:
        """Convert a database row to a Competency."""
        return Competency(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            discipline_id=row["discipline_id"],
            parent_id=row["parent_id"],
            coverage_target=row["coverage_target"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_example(row: sqlite3.Row) -> Example:
        """Convert a database row to an Example."""
        reviewed_at = row["reviewed_at"]
        if reviewed_at:
            reviewed_at = datetime.fromisoformat(reviewed_at)
        return Example(
            id=row["id"],
            question=row["question"],
            ideal_answer=row["ideal_answer"],
            competency_id=row["competency_id"],
            contributor_id=row["contributor_id"],
            discipline_id=row["discipline_id"],
            variants=json.loads(row["variants_json"]),
            context=row["context"],
            review_status=ReviewStatus(row["review_status"]),
            reviewed_by=row["reviewed_by"],
            reviewed_at=reviewed_at,
            is_test_set=bool(row["is_test_set"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_discipline_contributor(row: sqlite3.Row) -> DisciplineContributor:
        """Convert a database row to a DisciplineContributor."""
        return DisciplineContributor(
            discipline_id=row["discipline_id"],
            contributor_id=row["contributor_id"],
            role=ContributorRole(row["role"]),
            competency_area_ids=json.loads(row["competency_area_ids_json"]),
            joined_at=datetime.fromisoformat(row["joined_at"]),
        )

    @staticmethod
    def _row_to_curriculum_version(row: sqlite3.Row) -> CurriculumVersion:
        """Convert a database row to a CurriculumVersion."""
        return CurriculumVersion(
            id=row["id"],
            discipline_id=row["discipline_id"],
            version_number=row["version_number"],
            created_by=row["created_by"],
            example_count=row["example_count"],
            snapshot_json=row["snapshot_json"],
            status=CurriculumStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
