"""Lazy exports for builtin metric implementations.
Importing this package must stay side-effect free so runtime-scoped registry
assembly can safely reference builtin metric submodules after the global
registry has been frozen.
"""

from __future__ import annotations

import importlib

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ARCAGI2AccuracyMetric": ("gage_eval.metrics.builtin.arcagi2", "ARCAGI2AccuracyMetric"),
    "AppWorldDifficultyMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldDifficultyMetric"),
    "AppWorldFailCountMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldFailCountMetric"),
    "AppWorldPassCountMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldPassCountMetric"),
    "AppWorldSGCMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldSGCMetric"),
    "AppWorldSuccessMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldSuccessMetric"),
    "AppWorldTGCMetric": ("gage_eval.metrics.builtin.appworld", "AppWorldTGCMetric"),
    "CharXivReasoningMatchMetric": (
        "gage_eval.metrics.builtin.charxiv",
        "CharXivReasoningMatchMetric",
    ),
    "CompletionFlagMetric": ("gage_eval.metrics.builtin.arena", "CompletionFlagMetric"),
    "ContainsMatchMetric": ("gage_eval.metrics.builtin.text", "ContainsMatchMetric"),
    "DocVQAANLSMetric": ("gage_eval.metrics.builtin.docvqa_anls", "DocVQAANLSMetric"),
    "DrawFlagMetric": ("gage_eval.metrics.builtin.arena", "DrawFlagMetric"),
    "EpisodeDurationMsMetric": ("gage_eval.metrics.builtin.arena", "EpisodeDurationMsMetric"),
    "EpisodeLengthStepsMetric": ("gage_eval.metrics.builtin.arena", "EpisodeLengthStepsMetric"),
    "ExactMatchMetric": ("gage_eval.metrics.builtin.text", "ExactMatchMetric"),
    "FinalScorePerPlayerMetric": ("gage_eval.metrics.builtin.arena", "FinalScorePerPlayerMetric"),
    "GomokuAverageTurnsMetric": ("gage_eval.metrics.builtin.gomoku", "GomokuAverageTurnsMetric"),
    "GomokuIllegalRateMetric": ("gage_eval.metrics.builtin.gomoku", "GomokuIllegalRateMetric"),
    "GomokuWinRateMetric": ("gage_eval.metrics.builtin.gomoku", "GomokuWinRateMetric"),
    "IllegalActionCountMetric": ("gage_eval.metrics.builtin.arena", "IllegalActionCountMetric"),
    "IllegalReasonDistributionMetric": (
        "gage_eval.metrics.builtin.arena",
        "IllegalReasonDistributionMetric",
    ),
    "JudgeThresholdMetric": ("gage_eval.metrics.builtin.text", "JudgeThresholdMetric"),
    "LatencyMetric": ("gage_eval.metrics.builtin.text", "LatencyMetric"),
    "LegalActionRateMetric": ("gage_eval.metrics.builtin.arena", "LegalActionRateMetric"),
    "LikelihoodMetric": ("gage_eval.metrics.builtin.likelihood", "LikelihoodMetric"),
    "MMMUAccuracyMetric": ("gage_eval.metrics.builtin.mmmu", "MMMUAccuracyMetric"),
    "MathVistaAccuracyMetric": ("gage_eval.metrics.builtin.mathvista", "MathVistaAccuracyMetric"),
    "MultiChoiceAccuracyMetric": (
        "gage_eval.metrics.builtin.multi_choice",
        "MultiChoiceAccuracyMetric",
    ),
    "NumericMatchMetric": ("gage_eval.metrics.builtin.text", "NumericMatchMetric"),
    "ObsToActionLatencyMeanMetric": (
        "gage_eval.metrics.builtin.arena",
        "ObsToActionLatencyMeanMetric",
    ),
    "ObsToActionLatencyP50Metric": (
        "gage_eval.metrics.builtin.arena",
        "ObsToActionLatencyP50Metric",
    ),
    "ObsToActionLatencyP95Metric": (
        "gage_eval.metrics.builtin.arena",
        "ObsToActionLatencyP95Metric",
    ),
    "OnTimeRateMetric": ("gage_eval.metrics.builtin.arena", "OnTimeRateMetric"),
    "RankListMetric": ("gage_eval.metrics.builtin.arena", "RankListMetric"),
    "RankingMetric": ("gage_eval.metrics.builtin.ranking", "RankingMetric"),
    "RegexMatchMetric": ("gage_eval.metrics.builtin.text", "RegexMatchMetric"),
    "RetryCountMeanMetric": ("gage_eval.metrics.builtin.arena", "RetryCountMeanMetric"),
    "RetryCountP95Metric": ("gage_eval.metrics.builtin.arena", "RetryCountP95Metric"),
    "RewardPerSecondPerPlayerMetric": (
        "gage_eval.metrics.builtin.arena",
        "RewardPerSecondPerPlayerMetric",
    ),
    "ScoreMarginMetric": ("gage_eval.metrics.builtin.arena", "ScoreMarginMetric"),
    "ScreenSpotPointInBboxMetric": (
        "gage_eval.metrics.builtin.screenspot_pro",
        "ScreenSpotPointInBboxMetric",
    ),
    "SimpleQAVerifiedAccuracyMetric": (
        "gage_eval.metrics.builtin.simpleqa_verified",
        "SimpleQAVerifiedAccuracyMetric",
    ),
    "SimpleQAVerifiedJudgeAccuracyMetric": (
        "gage_eval.metrics.builtin.simpleqa_verified",
        "SimpleQAVerifiedJudgeAccuracyMetric",
    ),
    "Tau2AgentCostMetric": ("gage_eval.metrics.builtin.tau2", "Tau2AgentCostMetric"),
    "Tau2PassHatMetric": ("gage_eval.metrics.builtin.tau2", "Tau2PassHatMetric"),
    "Tau2PassMetric": ("gage_eval.metrics.builtin.tau2", "Tau2PassMetric"),
    "Tau2RewardMetric": ("gage_eval.metrics.builtin.tau2", "Tau2RewardMetric"),
    "Tau2UserCostMetric": ("gage_eval.metrics.builtin.tau2", "Tau2UserCostMetric"),
    "TerminationReasonMetric": ("gage_eval.metrics.builtin.arena", "TerminationReasonMetric"),
    "TextLengthMetric": ("gage_eval.metrics.builtin.text", "TextLengthMetric"),
    "TimeoutCountMetric": ("gage_eval.metrics.builtin.arena", "TimeoutCountMetric"),
    "WinFlagPerPlayerMetric": ("gage_eval.metrics.builtin.arena", "WinFlagPerPlayerMetric"),
    "WinnerPlayerIdMetric": ("gage_eval.metrics.builtin.arena", "WinnerPlayerIdMetric"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
from gage_eval.metrics.builtin.gomoku import (
    GomokuAverageTurnsMetric,
    GomokuIllegalRateMetric,
    GomokuWinRateMetric,
)
from gage_eval.metrics.builtin.multi_choice import MultiChoiceAccuracyMetric
from gage_eval.metrics.builtin.docvqa_anls import DocVQAANLSMetric
from gage_eval.metrics.builtin.mmmu import MMMUAccuracyMetric
from gage_eval.metrics.builtin.likelihood import LikelihoodMetric
from gage_eval.metrics.builtin.ranking import RankingMetric
from gage_eval.metrics.builtin.mathvista import MathVistaAccuracyMetric
from gage_eval.metrics.builtin.appworld import (
    AppWorldFailCountMetric,
    AppWorldPassCountMetric,
    AppWorldDifficultyMetric,
    AppWorldSGCMetric,
    AppWorldSuccessMetric,
    AppWorldTGCMetric,
)
from gage_eval.metrics.builtin.text import (
    ContainsMatchMetric,
    ExactMatchMetric,
    JudgeThresholdMetric,
    LatencyMetric,
    NumericMatchMetric,
    RegexMatchMetric,
    TextLengthMetric,
)
from gage_eval.metrics.builtin.tau2 import (
    Tau2AgentCostMetric,
    Tau2PassMetric,
    Tau2PassHatMetric,
    Tau2RewardMetric,
    Tau2UserCostMetric,
)
from gage_eval.metrics.builtin.simpleqa_verified import (
    SimpleQAVerifiedAccuracyMetric,
    SimpleQAVerifiedJudgeAccuracyMetric,
)
from gage_eval.metrics.builtin.arcagi2 import ARCAGI2AccuracyMetric
from gage_eval.metrics.builtin.charxiv import CharXivReasoningMatchMetric
from gage_eval.metrics.builtin.screenspot_pro import ScreenSpotPointInBboxMetric
from gage_eval.metrics.builtin.arena import (
    CompletionFlagMetric,
    DrawFlagMetric,
    EpisodeDurationMsMetric,
    EpisodeLengthStepsMetric,
    FinalScorePerPlayerMetric,
    IllegalActionCountMetric,
    IllegalReasonDistributionMetric,
    LegalActionRateMetric,
    ObsToActionLatencyMeanMetric,
    ObsToActionLatencyP50Metric,
    ObsToActionLatencyP95Metric,
    OnTimeRateMetric,
    RankListMetric,
    RetryCountMeanMetric,
    RetryCountP95Metric,
    RewardPerSecondPerPlayerMetric,
    ScoreMarginMetric,
    TerminationReasonMetric,
    TimeoutCountMetric,
    WinFlagPerPlayerMetric,
    WinnerPlayerIdMetric)
from gage_eval.metrics.builtin.inverse_ifeval import (
    InverseIFEvalJudgePassRateMetric,
    InverseIFEvalPassRateMetric,
)
from gage_eval.metrics.builtin.video_mme import VideoMMEAccuracyMetric

__all__ = [
    "ARCAGI2AccuracyMetric",
    "ExactMatchMetric",
    "ContainsMatchMetric",
    "NumericMatchMetric",
    "RegexMatchMetric",
    "JudgeThresholdMetric",
    "TextLengthMetric",
    "LatencyMetric",
    "GomokuWinRateMetric",
    "GomokuIllegalRateMetric",
    "GomokuAverageTurnsMetric",
    "MultiChoiceAccuracyMetric",
    "DocVQAANLSMetric",
    "MMMUAccuracyMetric",
    "MathVistaAccuracyMetric",
    "AppWorldTGCMetric",
    "AppWorldSGCMetric",
    "AppWorldSuccessMetric",
    "AppWorldPassCountMetric",
    "AppWorldFailCountMetric",
    "AppWorldDifficultyMetric",
    "LikelihoodMetric",
    "RankingMetric",
    "Tau2RewardMetric",
    "Tau2PassMetric",
    "Tau2PassHatMetric",
    "Tau2AgentCostMetric",
    "Tau2UserCostMetric",
    "SimpleQAVerifiedAccuracyMetric",
    "SimpleQAVerifiedJudgeAccuracyMetric",
    "ScreenSpotPointInBboxMetric",
    "CharXivReasoningMatchMetric",
    "FinalScorePerPlayerMetric",
    "EpisodeDurationMsMetric",
    "EpisodeLengthStepsMetric",
    "RewardPerSecondPerPlayerMetric",
    "ObsToActionLatencyMeanMetric",
    "ObsToActionLatencyP50Metric",
    "ObsToActionLatencyP95Metric",
    "OnTimeRateMetric",
    "TimeoutCountMetric",
    "LegalActionRateMetric",
    "IllegalActionCountMetric",
    "RetryCountMeanMetric",
    "RetryCountP95Metric",
    "IllegalReasonDistributionMetric",
    "WinnerPlayerIdMetric",
    "WinFlagPerPlayerMetric",
    "DrawFlagMetric",
    "RankListMetric",
    "ScoreMarginMetric",
    "TerminationReasonMetric",
    "CompletionFlagMetric",
    "InverseIFEvalJudgePassRateMetric",
    "InverseIFEvalPassRateMetric",
    "VideoMMEAccuracyMetric",
]
