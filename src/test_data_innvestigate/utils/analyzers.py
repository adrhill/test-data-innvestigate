"""
Lists and helper functions to iterate over all iNNvestigate analyzers.
"""

import innvestigate.analyzer as iAnalyzers
from innvestigate.analyzer import BoundedDeepTaylor
from innvestigate.analyzer import PatternNet

LRP_ANALYZERS = [
    "LRPZ",
    "LRPZIgnoreBias",
    "LRPEpsilon",
    "LRPEpsilonIgnoreBias",
    "LRPWSquare",
    "LRPFlat",
    "LRPAlpha2Beta1",
    "LRPAlpha2Beta1IgnoreBias",
    "LRPAlpha1Beta0",
    "LRPAlpha1Beta0IgnoreBias",
    "LRPZPlus",
    "LRPZPlusFast",
    "LRPSequentialPresetA",
    "LRPSequentialPresetB",
    "LRPSequentialPresetAFlat",
    "LRPSequentialPresetBFlat",
    "LRPSequentialPresetBFlatUntilIdx",
]

GRADIENT_BASED_ANALYZERS = [
    "Gradient",
    "BaselineGradient",
    # "InputTimesGradient",
    "Deconvnet",
    "GuidedBackprop",
    "IntegratedGradients",
    "SmoothGrad",
]

DEEP_TAYLOR_ANALYZERS = [
    "DeepTaylor",
    "BoundedDeepTaylor",
]

PATTERN_BASED_ANALYZERS = [
    "PatternNet",
    "PatternAttribution",
]

ANALYZERS = (
    LRP_ANALYZERS
    + GRADIENT_BASED_ANALYZERS
    + DEEP_TAYLOR_ANALYZERS
    + PATTERN_BASED_ANALYZERS
)


def get_analyzer_from_name(analyzer_name, model, patterns, input_range):
    """Similar to `create_analyzer` from iNNvestigate, but applies patterns and input_range."""
    Analyzer = getattr(iAnalyzers, analyzer_name)

    if issubclass(Analyzer, PatternNet):
        analyzer = Analyzer(model, patterns=patterns)
    elif issubclass(Analyzer, BoundedDeepTaylor):
        analyzer = Analyzer(model, low=input_range[0], high=input_range[1])
    else:
        analyzer = Analyzer(model)

    return analyzer
