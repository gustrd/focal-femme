"""
focal-femme: Automatic face clustering for photo collections.

A CLI tool that identifies the primary female subject in each image
and organizes photos by clustering similar faces.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .beauty_model import BeautyScorer
from .clusterer import FaceClusterer
from .detector import FaceDetector
from .renamer import FileRenamer
from .utils import ClusterState, FaceData

__all__ = [
    "BeautyScorer",
    "FaceClusterer",
    "FaceDetector",
    "FileRenamer",
    "ClusterState",
    "FaceData",
    "__version__",
]
