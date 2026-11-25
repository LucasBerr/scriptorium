"""Terminology: matching, enforcement, mining and the rendering ledger.

The term base is the project's central claim — that a term comes out the same way
on page 12 and on page 280 — so it is enforced by code rather than requested in a
prompt, and the enforcement is shared with the metric that scores it.
"""

from scriptorium.terminology.enforcement import (
    TerminologyViolation,
    accepted_forms,
    enforce,
    required_form,
)
from scriptorium.terminology.matching import count_occurrences, occurs, term_pattern
from scriptorium.terminology.mining import TermCandidate, mine_terms, tokenise
from scriptorium.terminology.morphology import inflections
from scriptorium.terminology.store import RenderingLedger, TerminologyStore

__all__ = [
    "RenderingLedger",
    "TermCandidate",
    "TerminologyStore",
    "TerminologyViolation",
    "accepted_forms",
    "count_occurrences",
    "enforce",
    "inflections",
    "mine_terms",
    "occurs",
    "required_form",
    "term_pattern",
    "tokenise",
]
