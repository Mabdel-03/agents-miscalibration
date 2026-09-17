"""Text grading (agent 04): private gold grading for MATH-500, MMLU-Pro and BBH-logic4, plus gold-blind helpers
(``normalize.option_identity`` / ``boolean_identity``, ``math_equiv.math_equivalent``) that selectors may import.

Math-Verify is imported lazily, so importing this package stays cheap.
"""
from __future__ import annotations

from agent_allocation.grading.text.grade import grade_text
from agent_allocation.grading.text.math_equiv import math_equivalent
from agent_allocation.grading.text.normalize import boolean_identity, option_identity

__all__ = ["boolean_identity", "grade_text", "math_equivalent", "option_identity"]
