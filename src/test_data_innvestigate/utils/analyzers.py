"""
Lists and helper functions to iterate over all iNNvestigate analyzers.
"""

from innvestigate.analyzer.deeptaylor import BoundedDeepTaylor
from innvestigate.analyzer.deeptaylor import DeepTaylor
from innvestigate.analyzer.gradient_based import BaselineGradient
from innvestigate.analyzer.gradient_based import Deconvnet
from innvestigate.analyzer.gradient_based import Gradient
from innvestigate.analyzer.gradient_based import GuidedBackprop
from innvestigate.analyzer.gradient_based import InputTimesGradient
from innvestigate.analyzer.gradient_based import IntegratedGradients
from innvestigate.analyzer.gradient_based import SmoothGrad
from innvestigate.analyzer.relevance_based.relevance_analyzer import BaselineLRPZ
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha1Beta0
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPAlpha1Beta0IgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha2Beta1
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPAlpha2Beta1IgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlphaBeta
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPEpsilon
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPEpsilonIgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPFlat
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetA,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetAFlat,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetB,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetBFlat,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetBFlatUntilIdx,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPWSquare
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZ
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZIgnoreBias
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZPlus
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZPlusFast
from innvestigate.analyzer.wrapper import AugmentReduceBase
from innvestigate.analyzer.wrapper import GaussianSmoother
from innvestigate.analyzer.wrapper import PathIntegrator

METHODS = {
    # Gradient based
    "Gradient": (Gradient, {}),
    "InputTimesGradient": (InputTimesGradient, {}),
    "Deconvnet": (Deconvnet, {}),
    "GuidedBackprop": (GuidedBackprop, {}),
    "IntegratedGradients": (IntegratedGradients, {}),
    "SmoothGrad": (SmoothGrad, {}),
    # Relevance based
    "LRPZ": (LRPZ, {}),
    "LRPZ_Flat_input_layer_rule": (LRPZ, {"input_layer_rule": "Flat"}),
    "LRPZ_boxed_input_layer_rule": (LRPZ, {"input_layer_rule": (-10, 10)}),
    "LRPZIgnoreBias": (LRPZIgnoreBias, {}),
    "LRPZPlus": (LRPZPlus, {}),
    "LRPZPlusFast": (LRPZPlusFast, {}),
    "BaselineLRPZ": (BaselineLRPZ, {}),
    "LRPAlpha1Beta0": (LRPAlpha1Beta0, {}),
    "LRPAlpha1Beta0IgnoreBias": (LRPAlpha1Beta0IgnoreBias, {}),
    "LRPAlpha2Beta1": (LRPAlpha2Beta1, {}),
    "LRPAlpha2Beta1IgnoreBias": (LRPAlpha2Beta1IgnoreBias, {}),
    "LRPEpsilon": (LRPEpsilon, {}),
    "LRPEpsilonIgnoreBias": (LRPEpsilonIgnoreBias, {}),
    "LRPFlat": (LRPFlat, {}),
    "LRPWSquare": (LRPWSquare, {}),
    "LRPSequentialPresetA": (LRPSequentialPresetA, {}),
    "LRPSequentialPresetB": (LRPSequentialPresetB, {}),
    "LRPSequentialPresetAFlat": (LRPSequentialPresetAFlat, {}),
    "LRPSequentialPresetBFlat": (LRPSequentialPresetBFlat, {}),
    # Deep taylor
    "DeepTaylor": (DeepTaylor, {}),
    "BoundedDeepTaylor": (BoundedDeepTaylor, {"low": -128, "high": 128}),
}
