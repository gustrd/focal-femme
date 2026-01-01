"""Tests for beauty_model module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from focal_femme.beauty_model import (
    BeautyModel,
    BeautyScorer,
    get_model_path,
)


class TestGetModelPath:
    def test_returns_path_in_cache_dir(self):
        path = get_model_path()
        assert isinstance(path, Path)
        assert ".cache" in str(path)
        assert "focal_femme" in str(path)
        assert path.name == "beauty_resnet18.pth"


class TestBeautyModel:
    def test_model_creation(self):
        model = BeautyModel()
        assert model is not None

    def test_model_forward_shape(self):
        model = BeautyModel()
        model.eval()

        # Create a dummy input (batch of 1, 3 channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Output should be a single value per image
        assert output.shape == (1, 1)

    def test_model_batch_forward(self):
        model = BeautyModel()
        model.eval()

        # Batch of 4 images
        dummy_input = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, 1)


class TestBeautyScorer:
    def test_scorer_initialization(self):
        scorer = BeautyScorer(device=torch.device('cpu'))
        assert scorer.device == torch.device('cpu')
        assert scorer._model is None  # Lazy initialization

    def test_preprocess_rgb_image(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Create a dummy RGB image (100x100)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor = scorer.preprocess(image)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_grayscale_image(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Create a dummy grayscale image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        tensor = scorer.preprocess(image)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_rgba_image(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Create a dummy RGBA image
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        tensor = scorer.preprocess(image)

        assert tensor.shape == (1, 3, 224, 224)

    @pytest.mark.slow
    def test_predict_returns_score(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Create a dummy face image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = scorer.predict(image)

        assert isinstance(score, float)
        assert 1.0 <= score <= 5.0

    def test_predict_empty_image_returns_zero(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Empty image
        image = np.array([])
        score = scorer.predict(image)

        assert score == 0.0

    def test_predict_small_image(self):
        scorer = BeautyScorer(device=torch.device('cpu'))

        # Very small image (should still work due to resize)
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        score = scorer.predict(image)

        assert isinstance(score, float)
