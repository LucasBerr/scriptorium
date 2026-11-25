"""Heuristic quality estimation.

**This is not a trained model and its weights are not evidence of anything.** It
exists for two reasons, both of which it serves honestly:

1. **Routing.** The loop needs a number to decide whether a segment is fit to
   ship, worth another pass, or needs a human. Something has to make that call
   before a trained model exists, and a transparent hand-weighted score is a
   better placeholder than a confident-looking one.
2. **Defining the feature set.** These signals — placeholder integrity, the count
   of terminology violations, the source/target length ratio, whether the output
   is just the input — are the features the phase-3 regressor learns to weight,
   trained on human post-edit distance. At that point the weights below are
   replaced by fitted ones and this module's score becomes a baseline to beat.

Until then, any report must label this as a heuristic. A hand-tuned number
presented as a quality measurement is exactly the sort of thing this project is
built to avoid.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from scriptorium.domain import GlossaryTerm
from scriptorium.segmentation.placeholders import check_tokens, tokens_in
from scriptorium.terminology.enforcement import enforce

# Hand-set, not fitted. Phase 3 replaces them.
PENALTY_BROKEN_PLACEHOLDER = 0.5
PENALTY_PER_VIOLATION = 0.25
PENALTY_MAX_VIOLATIONS = 0.5
PENALTY_MAX_LENGTH = 0.3
SCORE_IDENTICAL_TO_SOURCE = 0.4

# EN -> PT-BR runs roughly 15% longer. A rough constant is fine here: the
# penalty is for gross deviation, not for normal variation.
EXPECTED_LENGTH_RATIO = 1.15

_HAS_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


class QualitySignals(BaseModel):
    """The features behind the score. Logged so phase 3 can train on them."""

    placeholders_intact: bool = True
    terminology_violations: int = 0
    length_ratio: float = 0.0
    identical_to_source: bool = False
    is_empty: bool = False


class QualityEstimate(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    signals: QualitySignals
    heuristic: bool = True
    """Always true here. Phase 3's trained estimator sets it false."""


def estimate_quality(
    source: str,
    translation: str,
    *,
    glossary: Sequence[GlossaryTerm] = (),
) -> QualityEstimate:
    """Score a translated segment in [0, 1]. Both texts are placeholder-protected."""
    stripped = translation.strip()
    if not stripped:
        return QualityEstimate(score=0.0, signals=QualitySignals(is_empty=True))

    integrity = check_tokens(tokens_in(source), translation)
    violations = enforce(source, translation, glossary)
    ratio = len(stripped) / len(source.strip()) if source.strip() else 0.0

    # Identical output is usually an untranslated segment. "Usually": a short UI
    # string or a do-not-translate term can legitimately come back unchanged,
    # which is why this caps the score rather than zeroing it.
    identical = stripped == source.strip() and _HAS_LETTERS.search(source) is not None

    signals = QualitySignals(
        placeholders_intact=integrity.is_intact,
        terminology_violations=len(violations),
        length_ratio=ratio,
        identical_to_source=identical,
    )

    score = 1.0
    if not integrity.is_intact:
        score -= PENALTY_BROKEN_PLACEHOLDER
    score -= min(
        PENALTY_MAX_VIOLATIONS,
        PENALTY_PER_VIOLATION * len(violations),
    )
    deviation = abs(ratio - EXPECTED_LENGTH_RATIO) / EXPECTED_LENGTH_RATIO
    score -= min(PENALTY_MAX_LENGTH, PENALTY_MAX_LENGTH * deviation)

    if identical:
        score = min(score, SCORE_IDENTICAL_TO_SOURCE)

    return QualityEstimate(score=max(0.0, min(1.0, score)), signals=signals)
