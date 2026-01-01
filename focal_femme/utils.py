"""Utility functions for focal-femme: I/O, persistence, and helpers."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
