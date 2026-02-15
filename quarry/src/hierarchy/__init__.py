"""
Hierarchy module - Core of CHONK's document organization.

This module builds and analyzes document structure trees from extracted blocks.
"""

from chonk.hierarchy.tree import HierarchyNode, HierarchyTree
from chonk.hierarchy.builder import HierarchyBuilder
from chonk.hierarchy.analyzer import HierarchyAnalyzer

__all__ = [
    "HierarchyNode",
    "HierarchyTree",
    "HierarchyBuilder",
    "HierarchyAnalyzer",
]
