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
