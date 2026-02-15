"""
Test UI integration with backend diagnostic API.

This script:
1. Loads MIL-STD test data
2. Registers document with backend
3. Tests diagnostic API endpoints
4. Simulates the full UI workflow
"""

import json
import requests
from pathlib import Path

from chonk.core.document import (
    ChonkDocument,
    DocumentMetadata,
    Chunk,
    ChunkMetadata,
    QualityScore,
)
from chonk.server import _state

BASE_URL = "http://127.0.0.1:8420"


def load_test_document():
    """Load MIL-STD chunks from JSON."""
    json_path = Path('MIL-STD-extraction-chunks-HIERARCHY.json')
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    chunks_data = data['chunks'][:50]  # Use 50 chunks for testing
    chunks = []

    for chunk_data in chunks_data:
        quality = QualityScore(
            token_range=chunk_data['quality_details']['token_range'],
            sentence_complete=chunk_data['quality_details']['sentence_complete'],
            hierarchy_preserved=chunk_data['quality_details']['hierarchy_preserved'],
            table_integrity=chunk_data['quality_details']['table_integrity'],
            reference_complete=chunk_data['quality_details']['reference_complete'],
        )

        user_metadata = ChunkMetadata(
            tags=chunk_data['user_metadata']['tags'],
            hierarchy_hint=chunk_data['user_metadata']['hierarchy_hint'],
            notes=chunk_data['user_metadata']['notes'],
            custom=chunk_data['user_metadata']['custom'],
        )

        chunk = Chunk(
            id=chunk_data['id'],
            block_ids=chunk_data['block_ids'],
            content=chunk_data['content'],
            token_count=chunk_data['token_count'],
            hierarchy_path=chunk_data['hierarchy_path'],
            quality=quality,
            system_metadata=chunk_data['system_metadata'],
            user_metadata=user_metadata,
        )
        chunks.append(chunk)

    # Create document
    doc = ChonkDocument(
        id='test_mil_std',
        source_path=Path('MIL-STD-40051-2D.pdf'),
        source_type='pdf',
        blocks=[],
        chunks=chunks,
        metadata=DocumentMetadata(
            page_count=555,
            word_count=176294,
        ),
        loader_used='imported',
    )

    return doc


def register_document(doc):
    """Register document with backend."""
    # Add to _state documents
    if "documents" not in _state:
        _state["documents"] = {}
    _state["documents"][doc.id] = doc

    # Also add to current project's documents list
    if "current_project" in _state and _state["current_project"]:
        project = _state["current_project"]
        if doc not in project.documents:
            project.documents.append(doc)

    print(f"[*] Registered document: {doc.id} ({len(doc.chunks)} chunks)")


def test_diagnostic_api(doc_id):
    """Test the diagnostic API endpoints."""
    print("\n" + "=" * 80)
    print("TESTING DIAGNOSTIC API")
    print("=" * 80)

    # Test 1: Analyze document
    print("\n[1] Testing POST /api/diagnostics/analyze...")
    response = requests.post(
        f"{BASE_URL}/api/diagnostics/analyze",
        json={
            "document_id": doc_id,
            "include_questions": True,
            "test_questions": True,
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"    Success! Detected {len(result['problems'])} problems")
        print(f"    Statistics: {result['statistics']}")
        print(f"    Questions generated: {result['questions_generated']}")
        print(f"    Question pass rate: {result['question_pass_rate']:.1%}")
    else:
        print(f"    Error: {response.status_code} - {response.text}")
        return False

    # Test 2: Preview fixes
    print("\n[2] Testing POST /api/diagnostics/preview-fixes...")
    response = requests.post(
        f"{BASE_URL}/api/diagnostics/preview-fixes",
        json={
            "document_id": doc_id,
            "auto_resolve_conflicts": True,
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"    Success! Fix plan created")
        print(f"    Problems found: {result['problems_found']}")
        print(f"    Proposed fixes: {result['fix_plan']['total_actions']}")
        print(f"    Est. improvement: {result['fix_plan']['estimated_improvement'] * 100:.1f}%")

        if result['fix_plan']['warnings']:
            print(f"    Warnings:")
            for warning in result['fix_plan']['warnings']:
                print(f"      - {warning}")
    else:
        print(f"    Error: {response.status_code} - {response.text}")
        return False

    # Test 3: Apply fixes
    print("\n[3] Testing POST /api/diagnostics/apply-fixes...")
    response = requests.post(
        f"{BASE_URL}/api/diagnostics/apply-fixes",
        json={
            "document_id": doc_id,
            "auto_resolve_conflicts": True,
            "validate": True,
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"    Success! Fixes applied")
        print(f"    Chunks: {result['fix_result']['chunks_before']} -> {result['fix_result']['chunks_after']}")
        print(f"    Problems: {result['before']['problems']} -> {result['after']['problems']}")
        print(f"    Improvement: {result['improvement']['reduction_rate'] * 100:.1f}%")
    else:
        print(f"    Error: {response.status_code} - {response.text}")
        return False

    return True


def create_project():
    """Create a project on the backend."""
    response = requests.post(
        f"{BASE_URL}/api/project/new",
        json={"name": "Test Project"}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"    Created project: {result['name']}")
        return True
    else:
        print(f"    Error creating project: {response.status_code}")
        return False


def main():
    """Main test flow."""
    print("=" * 80)
    print("UI INTEGRATION TEST")
    print("=" * 80)

    # Step 0: Create project
    print("\n[STEP 0] Creating project...")
    if not create_project():
        print("Failed to create project")
        return

    # Step 1: Load test document
    print("\n[STEP 1] Loading test document...")
    doc = load_test_document()
    print(f"    Loaded {len(doc.chunks)} chunks from MIL-STD data")

    # Step 2: Register with backend
    print("\n[STEP 2] Registering document with backend...")
    register_document(doc)

    # Step 3: Test diagnostic API
    print("\n[STEP 3] Testing diagnostic API endpoints...")
    success = test_diagnostic_api(doc.id)

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS! All tests passed")
        print("=" * 80)
        print("\nThe UI should now be able to:")
        print("  1. Click 'RUN DIAGNOSTICS' to analyze chunks")
        print("  2. See detected problems in the dashboard")
        print("  3. Click 'PREVIEW AUTOMATIC FIXES' to see fix plan")
        print("  4. Click 'APPLY FIXES' to execute fixes")
        print("  5. See before/after metrics")
        print("\nDocument ID for UI: test_mil_std")
    else:
        print("\n" + "=" * 80)
        print("FAILED! Some tests did not pass")
        print("=" * 80)


if __name__ == '__main__':
    main()
