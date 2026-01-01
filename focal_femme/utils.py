"""Utility functions for focal-femme: I/O, persistence, and helpers."""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def setup_openvino_dll_path():
    """Add OpenVINO runtime libraries to DLL search path on Windows."""
    import sys
    import os
    if sys.platform != "win32":
        return

    # 1. Try to find openvino package via import
    try:
        # Import inside function to avoid circular/premature imports
        import openvino
        ov_base = Path(openvino.__file__).parent
        ov_libs = ov_base / "libs"
        if ov_libs.exists():
            os.add_dll_directory(str(ov_libs))
            logger.debug(f"Added OpenVINO libs to DLL search path: {ov_libs}")
            return
    except (ImportError, AttributeError):
        pass

    # 2. Fallback: Search in sys.path (site-packages)
    for p in sys.path:
        if "site-packages" in p:
            ov_libs = Path(p) / "openvino" / "libs"
            if ov_libs.exists():
                os.add_dll_directory(str(ov_libs))
                logger.debug(f"Added OpenVINO libs to DLL search path from sys.path: {ov_libs}")
                return


def get_best_device() -> torch.device:
    """
    Get the best available PyTorch device.

    Checks for availability in order of preference:
    1. CUDA (NVIDIA GPUs)
    2. XPU (Intel GPUs via intel-extension-for-pytorch)
    3. MPS (Apple Silicon)
    4. CPU (fallback)

    Returns:
        torch.device: The best available device
    """
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    # Check for XPU (Intel) - requires intel-extension-for-pytorch
    if hasattr(torch, "xpu"):
        if torch.xpu.is_available():
            device = torch.device("xpu")
            device_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel XPU"
            logger.debug(f"Using XPU device: {device_name}")
            return device
        else:
            logger.debug("XPU module found but no XPU device available")
    else:
        # Check if CPU-only PyTorch is installed
        if "+cpu" in torch.__version__:
            logger.debug(
                "CPU-only PyTorch installed. For Intel XPU support, reinstall PyTorch from Intel's index: "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu"
            )

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using Apple MPS device")
        return device

    # Fallback to CPU
    logger.debug("Using CPU device")
    return torch.device("cpu")


def get_onnx_providers() -> list[str]:
    """
    Get available ONNX Runtime execution providers in order of preference.

    Returns:
        List of provider names to try, best first
    """
    setup_openvino_dll_path()
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = []

    # Order of preference
    preferred_order = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",  # Intel optimized
        "DmlExecutionProvider",  # DirectML for Windows (AMD, Intel, NVIDIA)
        "CoreMLExecutionProvider",  # Apple
        "CPUExecutionProvider",
    ]

    for provider in preferred_order:
        if provider in available:
            providers.append(provider)

    # Ensure CPU is always available as fallback
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    logger.debug(f"ONNX providers: {providers}")
    return providers


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Default embeddings cache filename
EMBEDDINGS_CACHE_FILE = ".embeddings.pkl"


@dataclass
class FaceData:
    """Data structure for a detected face."""

    file_path: Path
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]  # (top, right, bottom, left)
    is_female: bool
    confidence: float = 0.0
    cluster_id: int | None = None
    beauty_score: float = 0.0


@dataclass
class ClusterState:
    """Persistent state for face clustering."""

    faces: dict[str, FaceData] = field(default_factory=dict)  # file_path_str -> FaceData
    next_cluster_id: int = 0
    version: int = 1

    def get_embeddings_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return embeddings as matrix and corresponding file paths."""
        if not self.faces:
            return np.array([]), []

        file_paths = list(self.faces.keys())
        embeddings = np.array([self.faces[fp].embedding for fp in file_paths])
        return embeddings, file_paths


def get_image_files(folder: Path) -> list[Path]:
    """Get all supported image files from a folder (non-recursive)."""
    if not folder.is_dir():
        raise ValueError(f"Not a valid directory: {folder}")

    files = []
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)

    return sorted(files)


def load_image(path: Path) -> np.ndarray:
    """Load an image file and return as RGB numpy array."""
    with Image.open(path) as img:
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)


def save_cluster_state(state: ClusterState, folder: Path) -> Path:
    """Save cluster state to pickle file in the target folder."""
    cache_path = folder / EMBEDDINGS_CACHE_FILE
    with open(cache_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path


def load_cluster_state(folder: Path) -> ClusterState | None:
    """Load cluster state from pickle file, or return None if not found."""
    cache_path = folder / EMBEDDINGS_CACHE_FILE
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
            if isinstance(state, ClusterState):
                return state
    except (pickle.UnpicklingError, EOFError, AttributeError):
        # Corrupted or incompatible cache file
        pass

    return None


def delete_cluster_state(folder: Path) -> bool:
    """Delete the cluster state cache file if it exists."""
    cache_path = folder / EMBEDDINGS_CACHE_FILE
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False


def format_cluster_id(cluster_id: int, beauty_score: int = 0) -> str:
    """Format cluster ID with beauty score (e.g., 'person_001_85')."""
    return f"person_{cluster_id:03d}_{beauty_score:02d}"


def parse_cluster_prefix(filename: str) -> tuple[str | None, str]:
    """
    Parse a filename to extract cluster prefix if present.

    Handles both old format (person_001_) and new format (person_001_85_).

    Returns:
        Tuple of (cluster_prefix or None, original_filename_without_prefix)
    """
    if filename.startswith("person_"):
        parts = filename.split("_", 3)
        if len(parts) >= 3 and parts[1].isdigit():
            # Check if third part is also a number (beauty score) - must be 2 digits
            if len(parts) >= 4 and parts[2].isdigit() and len(parts[2]) == 2:
                # New format: person_001_85_filename
                return f"person_{parts[1]}_{parts[2]}", parts[3]
            else:
                # Old format: person_001_filename
                # Rejoin everything after person_XXX_
                remainder = "_".join(parts[2:])
                return f"person_{parts[1]}", remainder

    return None, filename


def generate_safe_filename(
    original_path: Path,
    cluster_id: int,
    existing_files: set[str],
    beauty_score: int = 0,
) -> str:
    """
    Generate a safe filename with cluster prefix and beauty score.

    Args:
        original_path: Original file path
        cluster_id: Assigned cluster ID
        existing_files: Set of filenames already in use
        beauty_score: Normalized beauty score (0-99)

    Returns:
        New filename with cluster prefix and beauty score
    """
    prefix = format_cluster_id(cluster_id, beauty_score)

    # Strip any existing cluster prefix from the original name
    _, base_name = parse_cluster_prefix(original_path.name)

    # Build the new filename
    new_name = f"{prefix}_{base_name}"

    # Handle collisions by appending counter
    if new_name in existing_files:
        stem = original_path.stem
        suffix = original_path.suffix

        # Strip existing prefix from stem too
        _, clean_stem = parse_cluster_prefix(stem)

        counter = 1
        while True:
            new_name = f"{prefix}_{clean_stem}_{counter}{suffix}"
            if new_name not in existing_files:
                break
            counter += 1

    return new_name


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Calculate the area of a bounding box (top, right, bottom, left)."""
    top, right, bottom, left = bbox
    return (bottom - top) * (right - left)


def bbox_center_distance(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> float:
    """Calculate distance from bbox center to image center."""
    top, right, bottom, left = bbox

    bbox_center_x = (left + right) / 2
    bbox_center_y = (top + bottom) / 2

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    return np.sqrt(
        (bbox_center_x - image_center_x) ** 2 +
        (bbox_center_y - image_center_y) ** 2
    )
