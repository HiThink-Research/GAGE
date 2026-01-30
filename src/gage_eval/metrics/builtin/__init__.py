"""Built-in metric collection."""

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
]
