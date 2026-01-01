"""Tests for utils module."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from focal_femme.utils import (
    ClusterState,
    FaceData,
    bbox_area,
    bbox_center_distance,
    delete_cluster_state,
    format_cluster_id,
    generate_safe_filename,
    get_image_files,
    load_cluster_state,
    parse_cluster_prefix,
    save_cluster_state,
)


class TestBboxArea:
    def test_basic_area(self):
        # bbox is (top, right, bottom, left)
        bbox = (0, 100, 50, 0)  # 50 height x 100 width = 5000
        assert bbox_area(bbox) == 5000

    def test_zero_area(self):
        bbox = (10, 10, 10, 10)
        assert bbox_area(bbox) == 0

    def test_large_area(self):
        bbox = (0, 1920, 1080, 0)
        assert bbox_area(bbox) == 1920 * 1080


class TestBboxCenterDistance:
    def test_center_face(self):
        # Face at center of 100x100 image
        bbox = (25, 75, 75, 25)  # Center at (50, 50)
        distance = bbox_center_distance(bbox, 100, 100)
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_corner_face(self):
        # Face at corner
        bbox = (0, 20, 20, 0)  # Center at (10, 10)
        distance = bbox_center_distance(bbox, 100, 100)
        # Distance from (10, 10) to (50, 50)
        expected = np.sqrt(40**2 + 40**2)
        assert distance == pytest.approx(expected, abs=0.01)


class TestFormatClusterId:
    def test_single_digit(self):
        assert format_cluster_id(1) == "person_001"

    def test_double_digit(self):
        assert format_cluster_id(42) == "person_042"

    def test_triple_digit(self):
        assert format_cluster_id(123) == "person_123"

    def test_zero(self):
        assert format_cluster_id(0) == "person_000"


class TestParseClusterPrefix:
    def test_with_prefix(self):
        prefix, name = parse_cluster_prefix("person_001_IMG_1234.jpg")
        assert prefix == "person_001"
        assert name == "IMG_1234.jpg"

    def test_without_prefix(self):
        prefix, name = parse_cluster_prefix("IMG_1234.jpg")
        assert prefix is None
        assert name == "IMG_1234.jpg"

    def test_invalid_prefix_format(self):
        prefix, name = parse_cluster_prefix("person_abc_IMG_1234.jpg")
        assert prefix is None
        assert name == "person_abc_IMG_1234.jpg"

    def test_partial_prefix(self):
        prefix, name = parse_cluster_prefix("person_IMG_1234.jpg")
        assert prefix is None
        assert name == "person_IMG_1234.jpg"


class TestGenerateSafeFilename:
    def test_basic_rename(self):
        path = Path("/photos/IMG_1234.jpg")
        name = generate_safe_filename(path, 1, set())
        assert name == "person_001_IMG_1234.jpg"

    def test_collision_handling(self):
        path = Path("/photos/IMG_1234.jpg")
        existing = {"person_001_IMG_1234.jpg"}
        name = generate_safe_filename(path, 1, existing)
        assert name == "person_001_IMG_1234_1.jpg"

    def test_multiple_collisions(self):
        path = Path("/photos/IMG_1234.jpg")
        existing = {
            "person_001_IMG_1234.jpg",
            "person_001_IMG_1234_1.jpg",
            "person_001_IMG_1234_2.jpg",
        }
        name = generate_safe_filename(path, 1, existing)
        assert name == "person_001_IMG_1234_3.jpg"

    def test_strip_existing_prefix(self):
        path = Path("/photos/person_002_IMG_1234.jpg")
        name = generate_safe_filename(path, 1, set())
        assert name == "person_001_IMG_1234.jpg"


class TestClusterState:
    def test_empty_state(self):
        state = ClusterState()
        embeddings, paths = state.get_embeddings_matrix()
        assert embeddings.shape == (0,)
        assert paths == []

    def test_with_faces(self):
        state = ClusterState()
        state.faces["file1.jpg"] = FaceData(
            file_path=Path("file1.jpg"),
            embedding=np.array([1.0, 2.0, 3.0]),
            bbox=(0, 100, 100, 0),
            is_female=True,
        )
        state.faces["file2.jpg"] = FaceData(
            file_path=Path("file2.jpg"),
            embedding=np.array([4.0, 5.0, 6.0]),
            bbox=(0, 100, 100, 0),
            is_female=True,
        )

        embeddings, paths = state.get_embeddings_matrix()
        assert embeddings.shape == (2, 3)
        assert len(paths) == 2


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            state = ClusterState()
            state.faces["test.jpg"] = FaceData(
                file_path=Path("test.jpg"),
                embedding=np.array([1.0, 2.0, 3.0]),
                bbox=(0, 100, 100, 0),
                is_female=True,
                cluster_id=5,
            )
            state.next_cluster_id = 10

            save_cluster_state(state, folder)
            loaded = load_cluster_state(folder)

            assert loaded is not None
            assert len(loaded.faces) == 1
            assert loaded.next_cluster_id == 10
            assert "test.jpg" in loaded.faces
            assert loaded.faces["test.jpg"].cluster_id == 5

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            loaded = load_cluster_state(folder)
            assert loaded is None

    def test_delete_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            state = ClusterState()
            save_cluster_state(state, folder)

            assert delete_cluster_state(folder) is True
            assert delete_cluster_state(folder) is False  # Already deleted


class TestGetImageFiles:
    def test_finds_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # Create test files
            (folder / "image1.jpg").touch()
            (folder / "image2.png").touch()
            (folder / "document.txt").touch()
            (folder / "script.py").touch()

            files = get_image_files(folder)
            assert len(files) == 2
            names = {f.name for f in files}
            assert names == {"image1.jpg", "image2.png"}

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            files = get_image_files(folder)
            assert files == []

    def test_invalid_folder(self):
        with pytest.raises(ValueError):
            get_image_files(Path("/nonexistent/folder"))
