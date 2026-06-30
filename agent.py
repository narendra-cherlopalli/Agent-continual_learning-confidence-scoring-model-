"""
continual_learning/agent.py — Continual learning pattern.

Domain: IC match confidence scoring — a model that periodically retrains
on outcomes from its OWN past predictions, closing the loop between
action and improvement.

This is fundamentally different from rag_augmented/agent.py. RAG adds
RETRIEVABLE CONTEXT without changing any model — the underlying scoring
logic is static, only the context window grows. Continual learning
actually CHANGES the model's parameters based on accumulated outcomes —
specifically, whether a human approved or overrode each of the model's
past confidence-based suggestions.

The defining trait and the real engineering risk: a model that retrains
on its own outcomes can develop FEEDBACK LOOPS. If the model's
suggestions influence what a human reviews carefully versus rubber-stamps,
and the model then retrains on those same human decisions, errors can
compound rather than correct. This is tested explicitly below — a
scenario specifically constructed to surface whether the agent naively
trusts all outcomes equally or accounts for review depth.

This is intentionally a deterministic stand-in for what would be a real
ML retraining step (e.g. retraining an XGBoost classifier, as in
Close Command's confidence_scorer.py) — the retraining LOGIC here is a
weighted accuracy recalibration, not a literal gradient update. The
pattern this demonstrates — score, observe outcome, periodically
retrain, never let retraining happen mid-decision — is what transfers
to a real ML pipeline; the specific update rule does not need to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MIN_OUTCOMES_FOR_RETRAIN = 10


@dataclass
class MatchPrediction:
    match_id: str
    predicted_confidence: float  # 0.0-1.0, the model's confidence at prediction time
    suggested_status: str  # "MATCHED" | "EXCEPTION"


@dataclass
class MatchOutcome:
    match_id: str
    prediction: MatchPrediction
    human_decision: str  # "MATCHED" | "EXCEPTION"
    was_carefully_reviewed: bool  # did the human actually look closely, or rubber-stamp?

    @property
    def model_was_correct(self) -> bool:
        return self.prediction.suggested_status == self.human_decision


@dataclass
class RetrainResult:
    retrained: bool
    reason: str
    outcomes_used: int
    new_confidence_calibration: Optional[dict] = None
    accuracy_before: Optional[float] = None
    accuracy_after_recalibration: Optional[float] = None


class ContinualLearningMatchAgent:
    """
    Scores IC match confidence, accumulates outcomes from human review,
    and periodically retrains its confidence calibration. Critically:
    SCORING and RETRAINING are separate, non-overlapping operations.
    A score() call never triggers a retrain. retrain() never happens
    mid-prediction. This separation is what prevents a single noisy
    outcome from corrupting live predictions.
    """

    def __init__(self) -> None:
        self.confidence_calibration: dict[str, float] = {"base_threshold": 0.5}
        self.outcome_log: list[MatchOutcome] = []

    def score(self, match_id: str, raw_signal_strength: float) -> MatchPrediction:
        """
        Predict confidence and suggested status. Uses the CURRENT
        calibration — never affected by outcomes recorded after the
        last retrain() call, even if hundreds of outcomes have
        accumulated since then.
        """
        threshold = self.confidence_calibration["base_threshold"]
        confidence = min(1.0, max(0.0, raw_signal_strength))
        suggested_status = "MATCHED" if confidence >= threshold else "EXCEPTION"
        return MatchPrediction(
            match_id=match_id, predicted_confidence=round(confidence, 3), suggested_status=suggested_status,
        )

    def record_outcome(
        self, prediction: MatchPrediction, human_decision: str, was_carefully_reviewed: bool
    ) -> None:
        """Log what actually happened after a human reviewed the suggestion."""
        self.outcome_log.append(MatchOutcome(
            match_id=prediction.match_id, prediction=prediction,
            human_decision=human_decision, was_carefully_reviewed=was_carefully_reviewed,
        ))

    def retrain(self) -> RetrainResult:
        """
        Recalibrate the confidence threshold based on accumulated outcomes.
        Critically: outcomes from RUBBER-STAMPED reviews (was_carefully_reviewed=False)
        are EXCLUDED from retraining. A human clicking "approve" without
        actually checking is not evidence the model was right — including
        those outcomes would let the model's own influence on review
        behaviour feed back into its training data uncorrected, which
        compounds errors rather than correcting them.
        """
        if len(self.outcome_log) < MIN_OUTCOMES_FOR_RETRAIN:
            return RetrainResult(
                retrained=False,
                reason=f"Insufficient outcomes: {len(self.outcome_log)} (need {MIN_OUTCOMES_FOR_RETRAIN})",
                outcomes_used=0,
            )

        trustworthy_outcomes = [o for o in self.outcome_log if o.was_carefully_reviewed]

        if len(trustworthy_outcomes) < MIN_OUTCOMES_FOR_RETRAIN:
            return RetrainResult(
                retrained=False,
                reason=(
                    f"Insufficient CAREFULLY-REVIEWED outcomes: {len(trustworthy_outcomes)} of "
                    f"{len(self.outcome_log)} total (need {MIN_OUTCOMES_FOR_RETRAIN} carefully-reviewed). "
                    f"Rubber-stamped approvals are excluded — they are not reliable training signal."
                ),
                outcomes_used=0,
            )

        accuracy_before = self._compute_accuracy(trustworthy_outcomes)
        new_threshold = self._recalibrate_threshold(trustworthy_outcomes)

        old_calibration = dict(self.confidence_calibration)
        self.confidence_calibration["base_threshold"] = new_threshold

        # Re-score the SAME outcomes under the new calibration to report
        # whether the change actually improves accuracy — this is what
        # would gate a real deployment in production (don't ship a
        # retrain that makes things worse).
        accuracy_after = self._compute_accuracy_with_threshold(trustworthy_outcomes, new_threshold)

        if accuracy_after < accuracy_before:
            # Recalibration made things WORSE — roll back, don't ship it.
            self.confidence_calibration = old_calibration
            return RetrainResult(
                retrained=False,
                reason=f"Recalibration would have REDUCED accuracy ({accuracy_before:.2f} -> "
                       f"{accuracy_after:.2f}) — rolled back, did not apply.",
                outcomes_used=len(trustworthy_outcomes),
                accuracy_before=accuracy_before,
                accuracy_after_recalibration=accuracy_after,
            )

        return RetrainResult(
            retrained=True,
            reason=f"Recalibrated using {len(trustworthy_outcomes)} carefully-reviewed outcomes.",
            outcomes_used=len(trustworthy_outcomes),
            new_confidence_calibration=dict(self.confidence_calibration),
            accuracy_before=accuracy_before,
            accuracy_after_recalibration=accuracy_after,
        )

    @staticmethod
    def _compute_accuracy(outcomes: list[MatchOutcome]) -> float:
        if not outcomes:
            return 0.0
        correct = sum(1 for o in outcomes if o.model_was_correct)
        return round(correct / len(outcomes), 3)

    @staticmethod
    def _compute_accuracy_with_threshold(outcomes: list[MatchOutcome], threshold: float) -> float:
        """Re-evaluate what accuracy WOULD have been under a different threshold."""
        if not outcomes:
            return 0.0
        correct = 0
        for o in outcomes:
            re_predicted = "MATCHED" if o.prediction.predicted_confidence >= threshold else "EXCEPTION"
            if re_predicted == o.human_decision:
                correct += 1
        return round(correct / len(outcomes), 3)

    def _recalibrate_threshold(self, outcomes: list[MatchOutcome]) -> float:
        """
        Try a small grid of candidate thresholds, pick the one that would
        have produced the highest accuracy against the trustworthy outcome
        set. A real ML system would do gradient-based optimization; this
        grid search is the deterministic stand-in for that step.
        """
        candidates = [round(0.3 + 0.05 * i, 2) for i in range(15)]  # 0.30 to 1.00
        best_threshold = self.confidence_calibration["base_threshold"]
        best_accuracy = self._compute_accuracy_with_threshold(outcomes, best_threshold)

        for t in candidates:
            acc = self._compute_accuracy_with_threshold(outcomes, t)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t

        return best_threshold
