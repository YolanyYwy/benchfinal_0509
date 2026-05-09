"""Continual learning algorithms."""

from .base import CLAlgorithm, SequentialCL
from .ewc import EWCCL, EWCPPCL, OnlineEWCCL
from .fusion import AdaptiveFusionCL, ModelFusionCL
from .progressive import DynamicExpansionCL, ProgressiveNetsCL
from .replay import AdaptiveReplayCL, ReplayCL

__all__ = [
    "CLAlgorithm",
    "SequentialCL",
    "ReplayCL",
    "AdaptiveReplayCL",
    "EWCCL",
    "OnlineEWCCL",
    "EWCPPCL",
    "ProgressiveNetsCL",
    "DynamicExpansionCL",
    "ModelFusionCL",
    "AdaptiveFusionCL",
]
