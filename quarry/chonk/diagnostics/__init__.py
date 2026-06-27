"""
Diagnostic system for chunk quality analysis.

This module provides automated diagnostics for chunk problems:
- Static analysis (heuristics)
- Query-based testing (generated questions)
- Automatic fixes (merge/split strategies)
- Training data capture
"""

from chonk.diagnostics.analyzer import DiagnosticAnalyzer
from chonk.diagnostics.fix_orchestrator import FixOrchestrator
from chonk.diagnostics.question_generator import QuestionGenerator

__all__ = ["DiagnosticAnalyzer", "QuestionGenerator", "FixOrchestrator"]
