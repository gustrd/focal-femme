"""Tests for detector module."""

import numpy as np
import pytest

from focal_femme.detector import DetectedFace, FaceDetector


class TestDetectedFace:
    def test_creation(self):
        face = DetectedFace(
            bbox=(10, 110, 60, 10),
            embedding=np.array([1.0, 2.0, 3.0]),
            area=5000,
            center_distance=50.0,
            is_female=True,
            female_confidence=0.85,
            face_image=np.zeros((50, 100, 3), dtype=np.uint8),
        )

        assert face.bbox == (10, 110, 60, 10)
        assert face.area == 5000
        assert face.center_distance == 50.0
        assert face.is_female is True
        assert face.female_confidence == 0.85


class TestFaceDetector:
    def test_initialization(self):
        detector = FaceDetector()
        assert detector.female_threshold == 0.5

    def test_initialization_custom_threshold(self):
        detector = FaceDetector(female_threshold=0.7)
        assert detector.female_threshold == 0.7

    def test_select_primary_face_empty(self):
        detector = FaceDetector()
        result = detector.select_primary_face([])
        assert result is None

    def test_select_primary_face_no_females(self):
        detector = FaceDetector()
        faces = [
            DetectedFace(
                bbox=(10, 30, 30, 10),
                embedding=np.random.randn(512),
                area=400,
                center_distance=10.0,
                is_female=False,
                female_confidence=0.0,
                face_image=np.zeros((20, 20, 3), dtype=np.uint8),
            ),
        ]
        result = detector.select_primary_face(faces)
        assert result is None

    def test_select_primary_face_largest_selected(self):
        """Test that the largest female face is selected."""
        detector = FaceDetector()

        small_face = DetectedFace(
            bbox=(10, 30, 30, 10),
            embedding=np.random.randn(512),
            area=400,
            center_distance=10.0,
            is_female=True,
            female_confidence=0.9,
            face_image=np.zeros((20, 20, 3), dtype=np.uint8),
        )
        large_face = DetectedFace(
            bbox=(10, 110, 110, 10),
            embedding=np.random.randn(512),
            area=10000,
            center_distance=10.0,
            is_female=True,
            female_confidence=0.8,
            face_image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        result = detector.select_primary_face([small_face, large_face])

        assert result == large_face

    def test_select_primary_face_centrality_tiebreaker(self):
        """Test that centrality is used as tiebreaker for equal-sized faces."""
        detector = FaceDetector()

        corner_face = DetectedFace(
            bbox=(0, 100, 100, 0),
            embedding=np.random.randn(512),
            area=10000,
            center_distance=100.0,
            is_female=True,
            female_confidence=0.8,
            face_image=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        center_face = DetectedFace(
            bbox=(0, 100, 100, 0),
            embedding=np.random.randn(512),
            area=10000,
            center_distance=10.0,
            is_female=True,
            female_confidence=0.8,
            face_image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        result = detector.select_primary_face([corner_face, center_face])

        assert result == center_face


class TestFaceDetectorIntegration:
    """Integration tests that require PyTorch models.

    These tests are marked as slow and may be skipped in CI.
    """

    @pytest.mark.slow
    def test_process_image_no_faces(self, tmp_path):
        """Test processing an image with no faces."""
        detector = FaceDetector()

        # Create a simple blank image
        from PIL import Image

        img_path = tmp_path / "blank.jpg"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)

        result = detector.process_image(img_path)

        assert result is None

    @pytest.mark.slow
    def test_process_images_batch(self, tmp_path):
        """Test batch processing of images."""
        detector = FaceDetector()

        # Create multiple blank images
        from PIL import Image

        paths = []
        for i in range(3):
            img_path = tmp_path / f"blank_{i}.jpg"
            img = Image.new("RGB", (100, 100), color="white")
            img.save(img_path)
            paths.append(img_path)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        results = detector.process_images(paths, progress_callback)

        # No faces in blank images
        assert results == {}

        # Progress callback should have been called for each image
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.slow
    def test_detect_and_analyze_blank_image(self):
        """Test face detection on a blank image."""
        detector = FaceDetector()

        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect_and_analyze_faces(blank)

        assert faces == []
