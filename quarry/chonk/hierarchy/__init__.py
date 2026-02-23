"""
Hierarchy module - Core of CHONK's document organization.

This module builds and analyzes document structure trees from extracted blocks.
"""

from chonk.hierarchy.analyzer import HierarchyAnalyzer
from chonk.hierarchy.builder import HierarchyBuilder
from chonk.hierarchy.numbering import NumberingScheme, NumberingValidator
from chonk.hierarchy.tree import HierarchyNode, HierarchyTree

__all__ = [
    "HierarchyNode",
    "HierarchyTree",
    "HierarchyBuilder",
    "HierarchyAnalyzer",
    "NumberingScheme",
    "NumberingValidator",
]
