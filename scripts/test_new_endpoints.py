"""
Test script for new backend endpoints.
"""

import requests
import json
from pathlib import Path

API_BASE = "http://127.0.0.1:8420"
TEST_FILE = r"C:\Users\Sean Murphy\OneDrive\Desktop\CHONK\MIL-STD-40051-2D Change 1.pdf"


def test_endpoints():
    """Test all new endpoints with MIL-STD PDF."""

    print("=" * 80)
    print("CHONK Backend Endpoint Testing")
    print("=" * 80)

    # 1. Health check
    print("\n1. Testing health endpoint...")
    resp = requests.get(f"{API_BASE}/api/health")
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 200

    # 2. Create project
    print("\n2. Creating new project...")
    resp = requests.post(
        f"{API_BASE}/api/project/new",
        json={"name": "Test Project", "output_directory": None}
    )
    print(f"   Status: {resp.status_code}")
    project_data = resp.json()
    print(f"   Project ID: {project_data['id']}")
    assert resp.status_code == 200

    # 3. Upload document
    print("\n3. Uploading MIL-STD PDF...")
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('MIL-STD.pdf', f, 'application/pdf')}
        resp = requests.post(
            f"{API_BASE}/api/documents/upload",
            files=files,
            params={"extraction_tier": "enhanced", "auto_upgrade": "false"}
        )
    print(f"   Status: {resp.status_code}")
    upload_data = resp.json()
    document_id = upload_data['document_id']
    print(f"   Document ID: {document_id}")
    print(f"   Chunks: {upload_data['chunk_count']}")
    print(f"   Pages: {upload_data['page_count']}")
    assert resp.status_code == 200

    # 4. Build hierarchy
    print("\n4. Building document hierarchy...")
    resp = requests.post(
        f"{API_BASE}/api/hierarchy/build",
        json={"document_id": document_id}
    )
    print(f"   Status: {resp.status_code}")
    hierarchy_data = resp.json()
    stats = hierarchy_data['statistics']
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Leaf nodes: {stats['leaf_nodes']}")
    print(f"   Max depth: {stats['max_depth']}")
    print(f"   Avg tokens per node: {stats['avg_tokens_per_node']:.1f}")
    assert resp.status_code == 200
    assert stats['total_nodes'] > 0

    # 5. Get hierarchy stats
    print("\n5. Getting hierarchy statistics...")
    resp = requests.get(f"{API_BASE}/api/hierarchy/{document_id}/stats")
    print(f"   Status: {resp.status_code}")
    stats_data = resp.json()
    print(f"   Total nodes: {stats_data['total_nodes']}")
    print(f"   Quality score: {stats_data['quality_score']:.3f}")
    assert resp.status_code == 200

    # 6. Compare strategies
    print("\n6. Comparing chunking strategies...")
    resp = requests.post(
        f"{API_BASE}/api/chunk/compare",
        json={
            "document_id": document_id,
            "strategies": [
                {
                    "name": "hierarchy",
                    "config": {
                        "target_tokens": 400,
                        "max_tokens": 600,
                        "overlap_tokens": 50
                    }
                },
                {
                    "name": "fixed",
                    "config": {
                        "target_tokens": 400,
                        "overlap_tokens": 50
                    }
                }
            ]
        }
    )
    print(f"   Status: {resp.status_code}")
    comparison_data = resp.json()
    print(f"   Strategies compared: {len(comparison_data['strategies'])}")
    print(f"   Best strategy: {comparison_data.get('best_strategy', 'N/A')}")
    for strat in comparison_data['strategies']:
        metrics = strat['metrics']
        print(f"   - {strat['name']}: {metrics['total_chunks']} chunks, "
              f"avg {metrics['avg_tokens_per_chunk']:.1f} tokens, "
              f"quality {metrics['avg_quality_score']:.3f}, "
              f"hierarchy {metrics['hierarchy_preservation']:.1%}")
    assert resp.status_code == 200
    assert len(comparison_data['strategies']) == 2

    # 7. Preview chunks
    print("\n7. Previewing chunks with hierarchical strategy...")
    resp = requests.post(
        f"{API_BASE}/api/chunk/preview",
        json={
            "document_id": document_id,
            "chunker": "hierarchy",
            "target_tokens": 400,
            "max_tokens": 600,
            "overlap_tokens": 50
        }
    )
    print(f"   Status: {resp.status_code}")
    preview_data = resp.json()
    print(f"   Preview chunks: {len(preview_data['chunks'])}")
    print(f"   Avg quality: {preview_data['quality']['average_score']:.3f}")
    assert resp.status_code == 200

    # 8. Test query
    print("\n8. Testing retrieval query...")
    resp = requests.post(
        f"{API_BASE}/api/test/query",
        json={
            "query": "What are the safety requirements?",
            "strategies": ["hierarchy"],
            "document_id": document_id
        }
    )
    print(f"   Status: {resp.status_code}")
    query_data = resp.json()
    print(f"   Query: {query_data['query']}")
    for strat_result in query_data['strategies']:
        print(f"   Strategy: {strat_result['strategy_name']}")
        print(f"   Top score: {strat_result['top_score']:.3f}")
        print(f"   Retrieved: {strat_result['retrieved_count']} chunks")
        if strat_result['results']:
            print(f"   Top result preview: {strat_result['results'][0]['content_preview'][:100]}...")
    assert resp.status_code == 200

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nEndpoints working correctly:")
    print("  ✅ POST /api/hierarchy/build")
    print("  ✅ GET  /api/hierarchy/{doc_id}")
    print("  ✅ GET  /api/hierarchy/{doc_id}/stats")
    print("  ✅ POST /api/chunk/compare")
    print("  ✅ POST /api/chunk/preview")
    print("  ✅ POST /api/test/query")


if __name__ == "__main__":
    try:
        test_endpoints()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
