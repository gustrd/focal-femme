"""Face embedding clustering module using DBSCAN."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from .utils import ClusterState, FaceData

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    cluster_labels: dict[str, int]  # file_path -> cluster_id
    num_clusters: int
    num_noise: int  # Faces that didn't fit any cluster


class FaceClusterer:
    """Handles clustering of face embeddings using DBSCAN."""

    def __init__(
        self,
        eps: float = 0.4,
        min_samples: int = 2,
    ):
        """
        Initialize the clusterer.

        Args:
            eps: DBSCAN epsilon (cosine distance threshold, 0.3-0.5 typical)
            min_samples: DBSCAN min_samples parameter (minimum cluster size)
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, state: ClusterState) -> ClusterResult:
        """
        Perform DBSCAN clustering on face embeddings.

        Args:
            state: ClusterState containing face data

        Returns:
            ClusterResult with cluster assignments
        """
        embeddings, file_paths = state.get_embeddings_matrix()

        if len(file_paths) == 0:
            return ClusterResult(
                cluster_labels={},
                num_clusters=0,
                num_noise=0,
            )

        if len(file_paths) == 1:
            # Single face - assign it to cluster 0
            cluster_labels = {file_paths[0]: 0}
            return ClusterResult(
                cluster_labels=cluster_labels,
                num_clusters=1,
                num_noise=0,
            )

        # Debug: compute pairwise cosine distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
        # Get upper triangle (exclude diagonal)
        upper_tri = distances[np.triu_indices(len(embeddings), k=1)]
        logger.warning(f"Distance stats - min: {upper_tri.min():.3f}, max: {upper_tri.max():.3f}, mean: {upper_tri.mean():.3f}")

        # Run DBSCAN with cosine distance (better for face embeddings)
        # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        # Convert DBSCAN labels to our cluster IDs
        # DBSCAN uses -1 for noise, and 0, 1, 2, ... for clusters
        cluster_labels: dict[str, int] = {}
        unique_labels = set(labels)

        # Map DBSCAN labels to sequential cluster IDs
        # Noise points (-1) get unique individual cluster IDs
        label_mapping: dict[int, int] = {}
        next_cluster_id = state.next_cluster_id

        # First, map valid clusters
        for label in sorted(unique_labels):
            if label >= 0:
                label_mapping[label] = next_cluster_id
                next_cluster_id += 1

        # Now assign cluster IDs to each file
        num_noise = 0
        for file_path, label in zip(file_paths, labels):
            if label == -1:
                # Noise point - assign unique cluster ID
                cluster_labels[file_path] = next_cluster_id
                next_cluster_id += 1
                num_noise += 1
            else:
                cluster_labels[file_path] = label_mapping[label]

        # Update state with new next_cluster_id
        state.next_cluster_id = next_cluster_id

        # Count actual clusters (excluding noise)
        num_clusters = len([l for l in unique_labels if l >= 0])

        return ClusterResult(
            cluster_labels=cluster_labels,
            num_clusters=num_clusters,
            num_noise=num_noise,
        )

    def update_state_with_clusters(
        self,
        state: ClusterState,
        cluster_result: ClusterResult,
    ) -> None:
        """
        Update the ClusterState with cluster assignments.

        Args:
            state: ClusterState to update
            cluster_result: Result from clustering
        """
        for file_path, cluster_id in cluster_result.cluster_labels.items():
            if file_path in state.faces:
                state.faces[file_path].cluster_id = cluster_id

    def merge_new_faces(
        self,
        state: ClusterState,
        new_faces: dict[str, FaceData],
    ) -> list[str]:
        """
        Merge new face data into existing state.

        Args:
            state: Existing ClusterState
            new_faces: New face data to merge

        Returns:
            List of file paths that were added (not already in state)
        """
        added = []
        for file_path, face_data in new_faces.items():
            if file_path not in state.faces:
                state.faces[file_path] = face_data
                added.append(file_path)
            else:
                # Update existing entry (re-processed image)
                state.faces[file_path] = face_data
        return added

    def remove_missing_files(
        self,
        state: ClusterState,
        existing_files: set[str],
    ) -> list[str]:
        """
        Remove entries for files that no longer exist.

        Args:
            state: ClusterState to clean up
            existing_files: Set of file paths that currently exist

        Returns:
            List of file paths that were removed
        """
        removed = []
        to_remove = [fp for fp in state.faces if fp not in existing_files]

        for file_path in to_remove:
            del state.faces[file_path]
            removed.append(file_path)

        return removed


def get_cluster_summary(state: ClusterState) -> dict[int, list[str]]:
    """
    Get a summary of clusters and their member files.

    Args:
        state: ClusterState to summarize

    Returns:
        Dictionary mapping cluster_id to list of file paths
    """
    clusters: dict[int, list[str]] = {}

    for file_path, face_data in state.faces.items():
        if face_data.cluster_id is not None:
            if face_data.cluster_id not in clusters:
                clusters[face_data.cluster_id] = []
            clusters[face_data.cluster_id].append(file_path)

    return clusters


def get_cluster_beauty_scores(state: ClusterState) -> dict[int, float]:
    """
    Get the average beauty score for each cluster.

    Args:
        state: ClusterState containing face data with beauty scores

    Returns:
        Dictionary mapping cluster_id to average beauty score
    """
    cluster_scores: dict[int, list[float]] = {}

    for face_data in state.faces.values():
        if face_data.cluster_id is not None and face_data.beauty_score > 0:
            if face_data.cluster_id not in cluster_scores:
                cluster_scores[face_data.cluster_id] = []
            cluster_scores[face_data.cluster_id].append(face_data.beauty_score)

    # Compute averages
    return {
        cluster_id: sum(scores) / len(scores)
        for cluster_id, scores in cluster_scores.items()
        if scores
    }


def normalize_beauty_scores(beauty_scores: dict[int, float]) -> dict[int, int]:
    """
    Normalize beauty scores to 0-99 range using min-max normalization.

    Args:
        beauty_scores: Dictionary mapping cluster_id to average beauty score

    Returns:
        Dictionary mapping cluster_id to normalized score (0-99)
    """
    if not beauty_scores:
        return {}

    scores = list(beauty_scores.values())
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero if all scores are the same
    if max_score == min_score:
        # All same score - use 50 as middle value
        return {cluster_id: 50 for cluster_id in beauty_scores}

    # Normalize to 0-99 range
    normalized: dict[int, int] = {}
    for cluster_id, score in beauty_scores.items():
        normalized_value = (score - min_score) / (max_score - min_score) * 99
        normalized[cluster_id] = int(round(normalized_value))

    return normalized


def normalize_photo_beauty_scores(state: ClusterState) -> dict[str, int]:
    """
    Normalize individual photo beauty scores to 0-99 range across ALL photos.

    Unlike normalize_beauty_scores() which normalizes cluster averages,
    this normalizes each photo's beauty score globally.

    Args:
        state: ClusterState containing face data with beauty scores

    Returns:
        Dictionary mapping file_path (str) to normalized score (0-99)
    """
    if not state.faces:
        return {}

    # Collect all individual photo beauty scores (excluding zeros)
    photo_scores: dict[str, float] = {}
    for file_path, face_data in state.faces.items():
        if face_data.beauty_score > 0:
            photo_scores[file_path] = face_data.beauty_score

    if not photo_scores:
        return {}

    scores = list(photo_scores.values())
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero if all scores are the same
    if max_score == min_score:
        return {file_path: 50 for file_path in photo_scores}

    # Normalize to 0-99 range using min-max scaling
    normalized: dict[str, int] = {}
    for file_path, score in photo_scores.items():
        normalized_value = (score - min_score) / (max_score - min_score) * 99
        normalized[file_path] = int(round(normalized_value))

    # Handle photos with zero beauty scores
    for file_path, face_data in state.faces.items():
        if file_path not in normalized:
            normalized[file_path] = 0

    return normalized
