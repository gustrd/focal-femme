"""SCUT-FBP5500 facial beauty prediction module using ResNeXt-50."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .utils import get_best_device

logger = logging.getLogger(__name__)


class BeautyScorerSCUT:
    """SCUT-FBP5500 beauty prediction using ResNeXt-50."""

    def __init__(self, device: torch.device | None = None):
        """
        Initialize beauty scorer.

        Args:
            device: PyTorch device (cuda/xpu/mps/cpu or None for auto)
        """
        if device is None:
            self.device = get_best_device()
        else:
            self.device = device

        self._model = None

        # Preprocessing (from SCUT forward.py)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_model(self) -> nn.Module:
        """Lazy initialization of beauty model."""
        if self._model is None:
            # Use torchvision's ResNeXt-50
            model = models.resnext50_32x4d(weights=None)

            # Replace final FC layer for regression (1 output)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)

            # Load SCUT-FBP5500 pretrained weights
            model_path = self._get_model_path()

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                    # Handle different checkpoint formats
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # Load weights (strict=False to handle potential key mismatches)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded SCUT-FBP5500 ResNeXt-50 pretrained weights.")
                except Exception as e:
                    logger.warning(f"Could not load SCUT pretrained weights: {e}")
                    logger.info("Using ImageNet-pretrained ResNeXt-50 backbone only (less accurate beauty scores).")
            else:
                logger.warning(
                    f"SCUT-FBP5500 model not found at {model_path}. "
                    f"Using ImageNet-pretrained ResNeXt-50 backbone only (less accurate beauty scores)."
                )

            self._model = model.to(self.device)
            self._model.eval()

        return self._model

    def _get_model_path(self) -> Path:
        """Get path to model weights."""
        cache_dir = Path.home() / ".cache" / "focal_femme"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / "beauty_resnext50_scut.pth"

        if not model_path.exists():
            logger.info(
                f"SCUT-FBP5500 ResNeXt-50 model not found.\n"
                f"Download from: https://pan.baidu.com/s/1OhyJsCMfAdeo8kIZd29yAw\n"
                f"Extract 'ResNext50_All.pth' or similar and place at: {model_path}\n"
                f"Continuing with ImageNet-pretrained backbone only..."
            )

        return model_path

    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for SCUT model.

        Args:
            face_image: RGB face crop as numpy array

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        # Ensure correct data type
        if face_image.dtype != np.uint8:
            face_image = np.clip(face_image, 0, 255).astype(np.uint8)

        # Handle grayscale or RGBA images
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(face_image)

        # Apply transforms
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor

    def predict(self, face_image: np.ndarray) -> float:
        """
        Predict beauty score for a face.

        Args:
            face_image: RGB face crop as numpy array

        Returns:
            Beauty score in range 1.0-5.0 (or 0.0 for errors)
        """
        if face_image.size == 0:
            return 0.0

        try:
            model = self._get_model()
            tensor = self.preprocess(face_image).to(self.device)

            with torch.no_grad():
                output = model(tensor)
                score = output.item()

                # SCUT-FBP5500 outputs are in 1-5 range
                # Clamp to ensure valid range
                score = max(1.0, min(5.0, score))

            return score

        except Exception as e:
            logger.warning(f"Beauty prediction failed: {e}")
            return 0.0
