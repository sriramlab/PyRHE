from enum import Enum

class GenoImputeMethod(Enum):
    BINARY = "binary"
    MEAN = "mean"

class CovImputeMethod(Enum):
    IGNORE = "ignore"
    MEAN = "mean"

