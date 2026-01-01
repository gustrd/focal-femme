"""Tests for clusterer module."""

from pathlib import Path

import numpy as np
import pytest

from focal_femme.clusterer import FaceClusterer, get_cluster_summary
from focal_femme.utils import ClusterState, FaceData


def create_face_data(file_path: str, embedding: list[float]) -> FaceData:
    """Helper to create FaceData with minimal required fields."""
    return FaceData(
        file_path=Path(file_path),
        embedding=np.array(embedding),
        bbox=(0, 100, 100, 0),
        is_female=True,
    )


class TestFaceClusterer:
    def test_empty_state(self):
        state = ClusterState()
        clusterer = FaceClusterer()
        result = clusterer.cluster(state)

        assert result.cluster_labels == {}
        assert result.num_clusters == 0
        assert result.num_noise == 0

    def test_single_face(self):
        state = ClusterState()
        state.faces["file1.jpg"] = create_face_data("file1.jpg", [1.0, 0.0, 0.0])

        clusterer = FaceClusterer()
        result = clusterer.cluster(state)

        assert len(result.cluster_labels) == 1
        assert result.num_clusters == 1
        assert result.num_noise == 0

    def test_similar_faces_cluster_together(self):
        state = ClusterState()
        # Create three similar embeddings
        state.faces["file1.jpg"] = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        state.faces["file2.jpg"] = create_face_data("file2.jpg", [1.0, 0.1, 0.0])
        state.faces["file3.jpg"] = create_face_data("file3.jpg", [1.0, 0.0, 0.1])

        clusterer = FaceClusterer(eps=0.5, min_samples=2)
        result = clusterer.cluster(state)

        # All should be in the same cluster
        labels = list(result.cluster_labels.values())
        assert len(set(labels)) == 1
        assert result.num_clusters == 1

    def test_different_faces_separate_clusters(self):
        state = ClusterState()
        # Create very different embeddings
        state.faces["file1.jpg"] = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        state.faces["file2.jpg"] = create_face_data("file2.jpg", [1.0, 0.1, 0.0])
        state.faces["file3.jpg"] = create_face_data("file3.jpg", [0.0, 0.0, 10.0])
        state.faces["file4.jpg"] = create_face_data("file4.jpg", [0.0, 0.1, 10.0])

        clusterer = FaceClusterer(eps=0.5, min_samples=2)
        result = clusterer.cluster(state)

        # Should form 2 clusters
        assert result.num_clusters == 2

    def test_noise_points(self):
        state = ClusterState()
        # Two similar faces (will cluster)
        state.faces["file1.jpg"] = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        state.faces["file2.jpg"] = create_face_data("file2.jpg", [1.0, 0.1, 0.0])
        # One outlier (will be noise with min_samples=2)
        state.faces["file3.jpg"] = create_face_data("file3.jpg", [100.0, 100.0, 100.0])

        clusterer = FaceClusterer(eps=0.5, min_samples=2)
        result = clusterer.cluster(state)

        assert result.num_clusters == 1
        assert result.num_noise == 1

    def test_update_state_with_clusters(self):
        state = ClusterState()
        state.faces["file1.jpg"] = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        state.faces["file2.jpg"] = create_face_data("file2.jpg", [1.0, 0.1, 0.0])

        clusterer = FaceClusterer()
        result = clusterer.cluster(state)
        clusterer.update_state_with_clusters(state, result)

        # All faces should now have cluster IDs
        for face_data in state.faces.values():
            assert face_data.cluster_id is not None

    def test_merge_new_faces(self):
        state = ClusterState()
        state.faces["existing.jpg"] = create_face_data("existing.jpg", [1.0, 0.0, 0.0])

        new_faces = {
            "new1.jpg": create_face_data("new1.jpg", [0.0, 1.0, 0.0]),
            "new2.jpg": create_face_data("new2.jpg", [0.0, 0.0, 1.0]),
        }

        clusterer = FaceClusterer()
        added = clusterer.merge_new_faces(state, new_faces)

        assert len(added) == 2
        assert len(state.faces) == 3
        assert "new1.jpg" in state.faces
        assert "new2.jpg" in state.faces

    def test_remove_missing_files(self):
        state = ClusterState()
        state.faces["exists.jpg"] = create_face_data("exists.jpg", [1.0, 0.0, 0.0])
        state.faces["deleted.jpg"] = create_face_data("deleted.jpg", [0.0, 1.0, 0.0])

        existing_files = {"exists.jpg"}

        clusterer = FaceClusterer()
        removed = clusterer.remove_missing_files(state, existing_files)

        assert removed == ["deleted.jpg"]
        assert len(state.faces) == 1
        assert "exists.jpg" in state.faces


class TestGetClusterSummary:
    def test_empty_state(self):
        state = ClusterState()
        summary = get_cluster_summary(state)
        assert summary == {}

    def test_with_clusters(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0])
        face2.cluster_id = 0
        state.faces["file2.jpg"] = face2

        face3 = create_face_data("file3.jpg", [0.0, 0.0, 1.0])
        face3.cluster_id = 1
        state.faces["file3.jpg"] = face3

        summary = get_cluster_summary(state)

        assert len(summary) == 2
        assert len(summary[0]) == 2
        assert len(summary[1]) == 1

    def test_ignores_unclustered(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0])
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [0.0, 1.0, 0.0])
        face2.cluster_id = None
        state.faces["file2.jpg"] = face2

        summary = get_cluster_summary(state)

        assert len(summary) == 1
        assert 0 in summary
