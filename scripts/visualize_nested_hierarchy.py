"""
Visualize and explore the nested hierarchy structure.
"""
import json
from pathlib import Path

def main():
    file_path = Path("MIL-STD-extraction-NESTED-HIERARCHY.json")

    if not file_path.exists():
        print("ERROR: File not found!")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("NESTED HIERARCHY STRUCTURE")
    print("=" * 80)
    print()

    print("DOCUMENT STATS:")
    print(f"  Total sections: {data['hierarchy']['total_sections']}")
    print(f"  Leaf sections: {data['hierarchy']['leaf_sections']}")
    print(f"  Max depth: {data['hierarchy']['max_depth']}")
    print()

    print("STRUCTURE EXAMPLE:")
    print("-" * 80)
    print()
    print("Each section has this structure:")
    print()
    print("""
{
  "section_id": "E.5.3.5",              # Section number or identifier
  "heading": "Maintenance work packages...",  # The heading text
  "heading_block_id": "docling_blk_3783",     # Reference to heading block
  "heading_level": 1,                    # Heading level (1-6)
  "content": "Body text here...",        # Content WITHOUT heading
  "content_block_ids": ["docling_blk_3784", ...],  # References to content blocks
  "token_count": 290,                    # Total tokens (heading + content)
  "page_range": [344, 344],              # Pages this section spans
  "block_count": 3,                      # Total blocks
  "children": [                          # Nested subsections
    {
      "section_id": "E.5.3.5.1",
      "heading": "...",
      "content": "...",
      "children": []
    }
  ]
}
""")
    print()

    # Find a good example
    def find_example(sections, max_content=500):
        """Find a section with both content and children."""
        for section in sections:
            if (section['children'] and
                section['content'] and
                100 < len(section['content']) < max_content):
                return section
            if section['children']:
                result = find_example(section['children'], max_content)
                if result:
                    return result
        return None

    example = find_example(data['sections'])

    if example:
        print("REAL EXAMPLE:")
        print("-" * 80)
        print()
        print(f"Section ID: {example['section_id']}")
        print(f"Heading: {example['heading']}")
        print(f"Heading Level: {example['heading_level']}")
        print(f"Page Range: {example['page_range']}")
        print(f"Token Count: {example['token_count']}")
        print(f"Has {len(example['children'])} subsection(s)")
        print()
        print("Content (first 300 chars):")
        print(f"  {example['content'][:300]}...")
        print()
        print("Children:")
        for i, child in enumerate(example['children'][:5], 1):
            content_info = f"{len(child['content'])} chars" if child['content'] else "no content"
            children_info = f"{len(child['children'])} subsections" if child['children'] else "no subsections"
            print(f"  {i}. {child['section_id']}: {child['heading'][:50]}")
            print(f"     ({content_info}, {children_info})")
        if len(example['children']) > 5:
            print(f"  ... and {len(example['children']) - 5} more")
        print()

    print("=" * 80)
    print("HOW TO USE FOR RAG")
    print("=" * 80)
    print()
    print("Option 1: EMBED EACH SECTION SEPARATELY")
    print("-" * 80)
    print("""
# Each section becomes one embedding
for section in all_sections:
    text_to_embed = f\"\"\"{section['heading']}

{section['content']}\"\"\"

    metadata = {
        'section_id': section['section_id'],
        'heading': section['heading'],
        'page_range': section['page_range'],
        'has_subsections': len(section['children']) > 0
    }

    embed_and_store(text_to_embed, metadata)
""")
    print()

    print("Option 2: EMBED SECTIONS WITH CONTEXT")
    print("-" * 80)
    print("""
# Include parent heading for context
for section in all_sections:
    # Build context from parent hierarchy
    context = get_parent_headings(section)

    text_to_embed = f\"\"\"{context}

## {section['heading']}

{section['content']}\"\"\"

    embed_and_store(text_to_embed, metadata)
""")
    print()

    print("Option 3: EMBED WHOLE BRANCHES")
    print("-" * 80)
    print("""
# Embed section + all children (if small enough)
for section in all_sections:
    # Combine section with children
    full_text = combine_section_with_children(section, max_tokens=512)

    embed_and_store(full_text, metadata)
""")
    print()

    print("=" * 80)
    print("NAVIGATING THE HIERARCHY")
    print("=" * 80)
    print()
    print("The structure is a tree where:")
    print("  - Each section knows its heading and content")
    print("  - Parent-child relationships are explicit")
    print("  - You can traverse depth-first or breadth-first")
    print("  - Block IDs let you trace back to original extraction")
    print()

    # Statistics
    def count_sections_recursive(sections):
        """Count all sections recursively."""
        count = len(sections)
        for section in sections:
            count += count_sections_recursive(section['children'])
        return count

    def get_all_sections_flat(sections, result=None):
        """Flatten the tree into a list."""
        if result is None:
            result = []
        for section in sections:
            result.append(section)
            get_all_sections_flat(section['children'], result)
        return result

    all_sections = get_all_sections_flat(data['sections'])
    sections_with_content = [s for s in all_sections if s['content']]
    sections_with_children = [s for s in all_sections if s['children']]

    print("STATISTICS:")
    print(f"  Total sections: {len(all_sections)}")
    print(f"  Sections with content: {len(sections_with_content)} ({100*len(sections_with_content)/len(all_sections):.1f}%)")
    print(f"  Sections with children: {len(sections_with_children)} ({100*len(sections_with_children)/len(all_sections):.1f}%)")
    print()

    # Token distribution
    token_counts = [s['token_count'] for s in sections_with_content]
    if token_counts:
        print("TOKEN DISTRIBUTION (sections with content):")
        print(f"  Average: {sum(token_counts)/len(token_counts):.1f}")
        print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print()

        # Ranges
        ranges = [
            (0, 100, "0-100"),
            (101, 200, "101-200"),
            (201, 300, "201-300"),
            (301, 500, "301-500"),
            (501, 999999, "501+")
        ]
        print("  By range:")
        for min_t, max_t, label in ranges:
            count = sum(1 for t in token_counts if min_t <= t <= max_t)
            pct = 100 * count / len(token_counts)
            print(f"    {label:<10} {count:>4} sections ({pct:>5.1f}%)")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
