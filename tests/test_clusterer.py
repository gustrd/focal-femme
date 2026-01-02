"""Tests for clusterer module."""

from pathlib import Path

import numpy as np
import pytest

from focal_femme.clusterer import (
    FaceClusterer,
    get_cluster_summary,
    get_cluster_beauty_scores,
    normalize_beauty_scores,
    normalize_photo_beauty_scores,
)
from focal_femme.utils import ClusterState, FaceData


def create_face_data(
    file_path: str,
    embedding: list[float],
    beauty_score: float = 0.0,
) -> FaceData:
    """Helper to create FaceData with minimal required fields."""
    return FaceData(
        file_path=Path(file_path),
        embedding=np.array(embedding),
        bbox=(0, 100, 100, 0),
        is_female=True,
        beauty_score=beauty_score,
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
        state.faces["file2.jpg"] = create_face_data("file2.jpg", [0.95, 0.05, 0.0])
        # One outlier in opposite direction (will be noise with min_samples=2)
        # Cosine distance from [1,0,0] to [-1,0,0] is 2.0 (maximum)
        state.faces["file3.jpg"] = create_face_data("file3.jpg", [-1.0, 0.0, 0.0])

        clusterer = FaceClusterer(eps=0.3, min_samples=2)
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


class TestGetClusterBeautyScores:
    def test_empty_state(self):
        state = ClusterState()
        scores = get_cluster_beauty_scores(state)
        assert scores == {}

    def test_single_cluster(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=3.5)
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0], beauty_score=4.0)
        face2.cluster_id = 0
        state.faces["file2.jpg"] = face2

        scores = get_cluster_beauty_scores(state)

        assert len(scores) == 1
        assert scores[0] == pytest.approx(3.75, abs=0.01)

    def test_multiple_clusters(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=3.0)
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [0.0, 1.0, 0.0], beauty_score=4.5)
        face2.cluster_id = 1
        state.faces["file2.jpg"] = face2

        scores = get_cluster_beauty_scores(state)

        assert len(scores) == 2
        assert scores[0] == pytest.approx(3.0, abs=0.01)
        assert scores[1] == pytest.approx(4.5, abs=0.01)

    def test_ignores_zero_scores(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=3.5)
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0], beauty_score=0.0)
        face2.cluster_id = 0
        state.faces["file2.jpg"] = face2

        scores = get_cluster_beauty_scores(state)

        assert scores[0] == pytest.approx(3.5, abs=0.01)


class TestNormalizeBeautyScores:
    def test_empty_scores(self):
        normalized = normalize_beauty_scores({})
        assert normalized == {}

    def test_single_cluster(self):
        scores = {0: 3.5}
        normalized = normalize_beauty_scores(scores)
        assert normalized[0] == 50  # Single score gets middle value

    def test_two_clusters(self):
        scores = {0: 2.0, 1: 4.0}
        normalized = normalize_beauty_scores(scores)
        assert normalized[0] == 0
        assert normalized[1] == 99

    def test_three_clusters(self):
        scores = {0: 1.0, 1: 3.0, 2: 5.0}
        normalized = normalize_beauty_scores(scores)
        assert normalized[0] == 0
        assert normalized[1] == 50  # Middle value
        assert normalized[2] == 99

    def test_same_scores(self):
        scores = {0: 3.0, 1: 3.0, 2: 3.0}
        normalized = normalize_beauty_scores(scores)
        assert all(v == 50 for v in normalized.values())

    def test_close_scores(self):
        scores = {0: 3.0, 1: 3.5, 2: 4.0}
        normalized = normalize_beauty_scores(scores)
        assert normalized[0] == 0
        assert normalized[1] == 50
        assert normalized[2] == 99


class TestNormalizePhotoBeautyScores:
    def test_empty_state(self):
        state = ClusterState()
        normalized = normalize_photo_beauty_scores(state)
        assert normalized == {}

    def test_single_photo(self):
        state = ClusterState()
        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=3.5)
        state.faces["file1.jpg"] = face1

        normalized = normalize_photo_beauty_scores(state)
        assert normalized["file1.jpg"] == 50  # Single score gets middle value

    def test_multiple_photos_different_clusters(self):
        state = ClusterState()

        # Cluster 0 - two photos
        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=2.0)
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0], beauty_score=4.0)
        face2.cluster_id = 0
        state.faces["file2.jpg"] = face2

        # Cluster 1 - one photo
        face3 = create_face_data("file3.jpg", [0.0, 0.0, 1.0], beauty_score=6.0)
        face3.cluster_id = 1
        state.faces["file3.jpg"] = face3

        normalized = normalize_photo_beauty_scores(state)

        # Each photo should be normalized independently
        assert normalized["file1.jpg"] == 0  # Lowest score
        assert normalized["file2.jpg"] == 50  # Middle score
        assert normalized["file3.jpg"] == 99  # Highest score

    def test_same_scores(self):
        state = ClusterState()

        for i in range(3):
            face = create_face_data(f"file{i}.jpg", [1.0, 0.0, 0.0], beauty_score=3.0)
            state.faces[f"file{i}.jpg"] = face

        normalized = normalize_photo_beauty_scores(state)
        assert all(v == 50 for v in normalized.values())

    def test_ignores_zero_scores(self):
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=2.0)
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0], beauty_score=0.0)
        state.faces["file2.jpg"] = face2

        face3 = create_face_data("file3.jpg", [0.0, 0.0, 1.0], beauty_score=4.0)
        state.faces["file3.jpg"] = face3

        normalized = normalize_photo_beauty_scores(state)

        # Zero score should still be in result but set to 0
        assert normalized["file1.jpg"] == 0
        assert normalized["file2.jpg"] == 0
        assert normalized["file3.jpg"] == 99

    def test_two_photos_same_cluster(self):
        """Test that photos in same cluster get different scores in photo mode."""
        state = ClusterState()

        face1 = create_face_data("file1.jpg", [1.0, 0.0, 0.0], beauty_score=2.5)
        face1.cluster_id = 0
        state.faces["file1.jpg"] = face1

        face2 = create_face_data("file2.jpg", [1.0, 0.1, 0.0], beauty_score=4.5)
        face2.cluster_id = 0
        state.faces["file2.jpg"] = face2

        normalized = normalize_photo_beauty_scores(state)

        # Different photos should get different scores even in same cluster
        assert normalized["file1.jpg"] == 0
        assert normalized["file2.jpg"] == 99
