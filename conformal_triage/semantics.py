"""Plain-language meaning of radiomic features, for the "rich" and "semantic_only" prompts.

Each value is described relative to the training nodules (a percentile band), and for bands away
from typical a short phrase says what a higher or lower value means for the nodule. The phrases
describe what a feature measures, never which direction is malignant: linking e.g. an irregular
outline to malignancy is left to the language model.

Texture phrases follow the PyRadiomics documentation. Brightness is relative within each image
(images are min-max normalized one by one), which the prompt intro states once. Features whose
value mostly grows with nodule size (the non-normalized non-uniformity measures and the energies)
get no phrase, as do features without a clear plain-language reading.
"""

# Percentile bands (computed on the train split only) and their labels.
BANDS = [(10, "much lower than typical"), (30, "lower than typical"), (70, "close to typical"),
         (90, "higher than typical"), (101, "much higher than typical")]

# Names that read differently in everyday or medical language get their definition in the prompt.
DEFINITIONS = {
    "Elongation": "the ratio of the shortest to the longest axis; 1 means as wide as it is long",
    # TI-RADS-motivated features (conformal_triage/clinical.py)
    "TallerThanWideRatio": "vertical height divided by horizontal width; above 1 means taller than wide",
    "EchogenicityRatio": "mean brightness of the nodule divided by that of the surrounding tissue; "
                         "below 1 means darker than the surroundings, i.e. hypoechoic",
    "PunctateFociDensity": "small bright spots inside the nodule per 1,000 pixels, a proxy for punctate echogenic foci",
    "Solidity": "area divided by the area of its convex hull; 1 means no indentations in the outline",
    "MarginSharpness": "how abruptly brightness changes across the nodule border",
    "AnechoicFraction": "share of the nodule that is much darker than the surrounding tissue, as fluid is",
}

# feature name (last part of the PyRadiomics name, without 'Relative') -> (higher, lower)
MEANINGS = {
    # shape
    "MajorAxisLength": ("a longer nodule", "a shorter nodule"),
    "MinorAxisLength": ("a wider nodule across its short axis", "a narrower nodule across its short axis"),
    "MaximumDiameter": ("a larger maximum diameter", "a smaller maximum diameter"),
    "PixelSurface": ("a larger nodule area", "a smaller nodule area"),
    "MeshSurface": ("a larger nodule area", "a smaller nodule area"),
    "Perimeter": ("a longer outline", "a shorter outline"),
    "PerimeterSurfaceRatio": ("a longer outline relative to its area (a more irregular outline or a smaller nodule)",
                              "a shorter outline relative to its area (a smoother outline or a larger nodule)"),
    "Sphericity": ("a rounder, smoother outline", "a less round, more irregular outline"),
    "Elongation": ("a more compact, less elongated shape", "a more elongated, narrower shape"),
    # first order (brightness inside the nodule)
    "Mean": ("a brighter nodule on average", "a darker nodule on average"),
    "Median": ("a brighter nodule on average", "a darker nodule on average"),
    "RootMeanSquared": ("a brighter nodule on average", "a darker nodule on average"),
    "10Percentile": ("brighter dark areas", "darker dark areas"),
    "90Percentile": ("brighter bright areas", "dimmer bright areas"),
    "Minimum": ("brighter darkest spots", "darker darkest spots"),
    "Maximum": ("brighter brightest spots", "dimmer brightest spots"),
    "Range": ("a wider spread between the darkest and brightest spots inside the nodule",
              "a narrower spread between the darkest and brightest spots inside the nodule"),
    "InterquartileRange": ("more variable brightness inside the nodule", "more even brightness inside the nodule"),
    "MeanAbsoluteDeviation": ("more variable brightness inside the nodule", "more even brightness inside the nodule"),
    "RobustMeanAbsoluteDeviation": ("more variable brightness inside the nodule", "more even brightness inside the nodule"),
    "Variance": ("more variable brightness inside the nodule", "more even brightness inside the nodule"),
    "Entropy": ("more random, heterogeneous brightness", "more uniform brightness"),
    "Uniformity": ("more uniform brightness", "more random, heterogeneous brightness"),
    "Skewness": ("a brightness distribution with a tail of bright pixels",
                 "a brightness distribution with a tail of dark pixels"),
    "Kurtosis": ("a peaked brightness distribution with a few extreme pixels", "a flatter brightness distribution"),
    # GLCM (pairs of neighboring pixels)
    "Contrast": ("larger brightness differences between neighboring pixels",
                 "smaller brightness differences between neighboring pixels"),
    "DifferenceAverage": ("larger brightness differences between neighboring pixels",
                          "smaller brightness differences between neighboring pixels"),
    "DifferenceEntropy": ("more random differences between neighboring pixels",
                          "more regular differences between neighboring pixels"),
    "DifferenceVariance": ("more variable differences between neighboring pixels",
                           "more consistent differences between neighboring pixels"),
    "Id": ("a smoother, more homogeneous texture", "a rougher, less homogeneous texture"),
    "Idm": ("a smoother, more homogeneous texture", "a rougher, less homogeneous texture"),
    "Idn": ("a smoother, more homogeneous texture", "a rougher, less homogeneous texture"),
    "Idmn": ("a smoother, more homogeneous texture", "a rougher, less homogeneous texture"),
    "InverseVariance": ("a smoother, more homogeneous texture", "a rougher, less homogeneous texture"),
    "JointEnergy": ("a few dominant local patterns (a more uniform texture)",
                    "many different local patterns (a more varied texture)"),
    "MaximumProbability": ("one dominant local pattern (a more uniform texture)",
                           "no single dominant local pattern (a more varied texture)"),
    "JointEntropy": ("a more random, complex texture", "a simpler, more regular texture"),
    "SumEntropy": ("a more random, complex texture", "a simpler, more regular texture"),
    "Correlation": ("more predictable brightness between neighboring pixels",
                    "less predictable brightness between neighboring pixels"),
    "MCC": ("more predictable brightness between neighboring pixels",
            "less predictable brightness between neighboring pixels"),
    "Imc2": ("more predictable brightness between neighboring pixels",
             "less predictable brightness between neighboring pixels"),
    "Autocorrelation": ("brighter neighboring pixel pairs", "darker neighboring pixel pairs"),
    "JointAverage": ("brighter neighboring pixel pairs", "darker neighboring pixel pairs"),
    "SumAverage": ("brighter neighboring pixel pairs", "darker neighboring pixel pairs"),
    "ClusterTendency": ("more variable brightness across neighboring pixels",
                        "more even brightness across neighboring pixels"),
    "SumSquares": ("more variable brightness across neighboring pixels",
                   "more even brightness across neighboring pixels"),
    "ClusterProminence": ("a more asymmetric brightness distribution", "a more symmetric brightness distribution"),
    # GLSZM (patches: connected areas of the same brightness)
    "SmallAreaEmphasis": ("a finer texture with more small patches", "a coarser texture with fewer small patches"),
    "LargeAreaEmphasis": ("a coarser texture with larger patches", "a finer texture with smaller patches"),
    "ZonePercentage": ("a finer texture made of many small patches", "a coarser texture made of fewer, larger patches"),
    "ZoneVariance": ("more variable patch sizes", "more similar patch sizes"),
    "ZoneEntropy": ("a more heterogeneous, irregular texture", "a more homogeneous, regular texture"),
    "SizeZoneNonUniformityNormalized": ("more variable patch sizes", "more similar patch sizes"),
    "LowGrayLevelZoneEmphasis": ("more dark patches", "fewer dark patches"),
    "HighGrayLevelZoneEmphasis": ("more bright patches", "fewer bright patches"),
    "SmallAreaLowGrayLevelEmphasis": ("more small, dark patches inside the nodule", "fewer small, dark patches inside the nodule"),
    "SmallAreaHighGrayLevelEmphasis": ("more small, bright patches inside the nodule", "fewer small, bright patches inside the nodule"),
    "LargeAreaLowGrayLevelEmphasis": ("more large, dark patches inside the nodule", "fewer large, dark patches inside the nodule"),
    "LargeAreaHighGrayLevelEmphasis": ("more large, bright patches inside the nodule", "fewer large, bright patches inside the nodule"),
    # GLRLM (runs: straight lines of the same brightness)
    "ShortRunEmphasis": ("a finer texture with more short runs", "a coarser texture with fewer short runs"),
    "LongRunEmphasis": ("a coarser texture with longer runs of the same brightness",
                        "a finer texture with shorter runs of the same brightness"),
    "RunPercentage": ("a finer texture", "a coarser texture"),
    "RunVariance": ("more variable run lengths", "more similar run lengths"),
    "RunEntropy": ("a more heterogeneous, irregular texture", "a more homogeneous, regular texture"),
    "RunLengthNonUniformityNormalized": ("more variable run lengths", "more similar run lengths"),
    "LowGrayLevelRunEmphasis": ("more dark areas in the texture", "fewer dark areas in the texture"),
    "HighGrayLevelRunEmphasis": ("more bright areas in the texture", "fewer bright areas in the texture"),
    # GLDM (dependence: how many neighbors share a pixel's brightness)
    "SmallDependenceEmphasis": ("a more heterogeneous texture (few similar neighbors)",
                                "a more homogeneous texture (many similar neighbors)"),
    "LargeDependenceEmphasis": ("a more homogeneous texture (many similar neighbors)",
                                "a more heterogeneous texture (few similar neighbors)"),
    "DependenceVariance": ("a mix of uniform and non-uniform regions", "a more consistent local texture"),
    "DependenceEntropy": ("a more heterogeneous, irregular texture", "a more homogeneous, regular texture"),
    "DependenceNonUniformityNormalized": ("a less homogeneous local texture pattern", "a more homogeneous local texture pattern"),
    "LowGrayLevelEmphasis": ("more dark areas in the texture", "fewer dark areas in the texture"),
    "HighGrayLevelEmphasis": ("more bright areas in the texture", "fewer bright areas in the texture"),
    # GLSZM / GLRLM / GLDM gray-level variability (normalized)
    "GrayLevelVariance": ("more variable brightness across the texture", "more even brightness across the texture"),
    "GrayLevelNonUniformityNormalized": ("more uniform brightness across the texture",
                                         "more varied brightness across the texture"),
    # TI-RADS-motivated features (clinical.py); phrases use TI-RADS vocabulary, still without risk
    "TallerThanWideRatio": ("a taller, more vertically oriented nodule", "a wider, more horizontally oriented nodule"),
    "EchogenicityRatio": ("a nodule closer in brightness to the surrounding tissue (less hypoechoic)",
                          "a nodule darker than the surrounding tissue (more hypoechoic)"),
    "PunctateFociDensity": ("more punctate echogenic foci", "fewer punctate echogenic foci"),
    "Solidity": ("a smoother, more convex outline", "a more lobulated or irregular outline"),
    "MarginSharpness": ("a sharper, well-defined margin", "a blurrier, less well-defined margin"),
    "AnechoicFraction": ("a larger fluid-like (cystic) part", "a more solid composition"),
    # NGTDM (difference from the neighborhood average)
    "Coarseness": ("a coarser texture with slow brightness changes", "a finer texture with rapid brightness changes"),
    "Busyness": ("rapid brightness changes between neighborhoods", "slow brightness changes between neighborhoods"),
    "Complexity": ("a more complex, busy texture", "a simpler texture"),
    "Strength": ("a clearer, more visible texture pattern", "a faint texture pattern"),
}


def feature_key(raw_name: str) -> tuple[str, bool]:
    """'original_shape2D_MajorAxisLengthRelative' -> ('MajorAxisLength', True)."""
    name = raw_name.split("_")[-1]
    relative = name.endswith("Relative")
    return (name[: -len("Relative")] if relative else name), relative


def band(percentile: float) -> tuple[int, str]:
    """Index (0 = much lower ... 4 = much higher) and label of a percentile (0-100)."""
    for i, (upper, label) in enumerate(BANDS):
        if percentile < upper:
            return i, label
    return len(BANDS) - 1, BANDS[-1][1]


def meaning(raw_name: str, percentile: float) -> str | None:
    """Phrase for a value away from typical, or None (typical band, or no plain-language reading)."""
    key, relative = feature_key(raw_name)
    i, _ = band(percentile)
    if i == 2 or key not in MEANINGS:
        return None
    phrase = MEANINGS[key][0 if i > 2 else 1]
    # Sizes divided by the image scale; the perimeter/area ratio is already a shape measure.
    return phrase + " relative to the image size" if relative and key != "PerimeterSurfaceRatio" else phrase
