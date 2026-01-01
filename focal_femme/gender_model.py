"""Gender classification using ONNX model."""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests

from .utils import get_onnx_providers

logger = logging.getLogger(__name__)

# Gender model configuration
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx"
MODEL_FILENAME = "gender_googlenet.onnx"


def get_model_path() -> Path:
    """Get the path to the gender model, downloading if necessary."""
    # Store in user's cache directory
    cache_dir = Path.home() / ".cache" / "focal-femme"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / MODEL_FILENAME

    if not model_path.exists():
        logger.info(f"Downloading gender model to {model_path}...")
        try:
            response = requests.get(MODEL_URL, timeout=60)
            response.raise_for_status()
            model_path.write_bytes(response.content)
            logger.info("Gender model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download gender model: {e}") from e

    return model_path


class GenderClassifier:
    """Classifies gender from face images using ONNX model."""

    def __init__(self):
        """Initialize the gender classifier."""
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def _get_session(self) -> ort.InferenceSession:
        """Lazy initialization of ONNX session."""
        if self._session is None:
            model_path = get_model_path()

            # Create ONNX inference session with best available providers
            providers = get_onnx_providers()
            self._session = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.debug(f"ONNX session using: {self._session.get_providers()}")

        return self._session

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the gender model.

        Args:
            face_image: RGB face crop

        Returns:
            Preprocessed tensor
        """
        # Resize to model input size (224x224 for GoogleNet)
        resized = cv2.resize(face_image, (224, 224))

        # Convert RGB to BGR (model expects BGR)
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        # Normalize (ImageNet mean subtraction)
        mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        normalized = bgr.astype(np.float32) - mean

        # Transpose to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def predict(self, face_image: np.ndarray) -> tuple[bool, float]:
        """
        Predict gender from face image.

        Args:
            face_image: RGB face crop as numpy array

        Returns:
            Tuple of (is_female, confidence)
            - is_female: True if classified as female
            - confidence: Probability of being female (0-1)
        """
        if face_image.size == 0:
            return False, 0.0

        try:
            session = self._get_session()

            # Preprocess
            input_tensor = self.preprocess(face_image)

            # Run inference
            outputs = session.run(None, {self._input_name: input_tensor})

            # Get probabilities
            # Output is [batch, 2] where index 0 = male, 1 = female
            probs = outputs[0][0]

            # Apply softmax if not already applied
            if probs.min() < 0 or probs.max() > 1:
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()

            female_prob = float(probs[1])
            is_female = female_prob > 0.5

            return is_female, female_prob

        except Exception as e:
            logger.warning(f"Gender prediction failed: {e}")
            return False, 0.0
