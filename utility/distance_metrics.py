# utility/distance_metrics.py
from enum import Enum


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    INNER_PRODUCT = "ip"


class StoringMethod(str, Enum):
    CHROMA = "chroma"
    SIMPLE = "simple"
    LLAMA_INDEX = "llama_index"
