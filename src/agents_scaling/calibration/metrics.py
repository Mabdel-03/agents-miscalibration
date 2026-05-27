"""Calibration metrics: ECE, MCE, Brier score, reliability bins.

Definitions follow Guo et al. 2017 (*On Calibration of Modern Neural Networks*,
arXiv:1706.04599). ECE uses equal-width confidence bins (default 15). Each datapoint is
(confidence in [0,1], correct in {0,1}); confidence is the model's stated p(its answer
is correct), so calibration asks whether confidence matches empirical accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    avg_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.avg_confidence)


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    n: int
    accuracy: float
    avg_confidence: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "n": self.n,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "bins": [vars(b) for b in self.bins],
        }


def compute_calibration(
    confidences: list[float], corrects: list[bool], n_bins: int = 15
) -> CalibrationReport:
    """ECE/MCE/Brier + reliability bins for paired (confidence, correct) data.

    ECE = sum over bins of (n_b / N) * |acc_b - conf_b|.
    MCE = max over non-empty bins of |acc_b - conf_b|.
    Brier = mean (confidence - correct)^2.
    """
    if len(confidences) != len(corrects):
        raise ValueError("confidences and corrects must have equal length")
    n = len(confidences)
    if n == 0:
        return CalibrationReport(0.0, 0.0, 0.0, 0, 0.0, 0.0, [])

    y = [1.0 if c else 0.0 for c in corrects]
    accuracy = sum(y) / n
    avg_conf = sum(confidences) / n
    brier = sum((p - t) ** 2 for p, t in zip(confidences, y)) / n

    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    width = 1.0 / n_bins
    for b in range(n_bins):
        lo = b * width
        hi = (b + 1) * width
        # Membership is [lo, hi); the last bin also takes the right edge so that a
        # confidence of exactly 1.0 is counted.
        in_bin = [
            i
            for i, p in enumerate(confidences)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)
        ]
        if not in_bin:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        cnt = len(in_bin)
        conf_b = sum(confidences[i] for i in in_bin) / cnt
        acc_b = sum(y[i] for i in in_bin) / cnt
        gap = abs(acc_b - conf_b)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lo, hi, cnt, conf_b, acc_b))

    return CalibrationReport(
        ece=ece, mce=mce, brier=brier, n=n, accuracy=accuracy, avg_confidence=avg_conf, bins=bins
    )
