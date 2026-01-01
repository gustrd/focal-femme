"""Facial beauty prediction module using ResNet18."""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .utils import get_best_device

logger = logging.getLogger(__name__)

# Google Drive file ID for pretrained beauty model
# From: https://github.com/etrain-xyz/facial-beauty-prediction
GDRIVE_FILE_ID = "1-JGQ1B9w6dteDHJPNwp-YWDGMcPO16LL"
MODEL_FILENAME = "beauty_resnet18.pth"


def get_model_path() -> Path:
    """Get the path where the beauty model should be stored."""
    cache_dir = Path.home() / ".cache" / "focal-femme"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / MODEL_FILENAME


def download_from_gdrive(file_id: str, destination: Path) -> bool:
    """
    Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Path to save the file

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        import requests

        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        session = requests.Session()
        response = session.get(url, stream=True)

        # Check for confirmation token (for large files)
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(url, stream=True)
                break

        # Check if we got an HTML page (error) instead of the file
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            logger.warning("Google Drive download may require manual confirmation")
            return False

        # Save the file
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        return True

    except Exception as e:
        logger.warning(f"Failed to download beauty model: {e}")
        return False


class BeautyModel(nn.Module):
    """ResNet18-based facial beauty prediction model."""

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace the final fully connected layer for regression
        # The SCUT-FBP5500 ResNet18 weights appear to use a single Linear layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BeautyScorer:
    """Handles facial beauty prediction."""

    def __init__(self, device: torch.device | None = None):
        """
        Initialize the beauty scorer.

        Args:
            device: Torch device ('cuda', 'cpu', 'xpu', 'mps', or None for auto)
        """
        if device is None:
            self.device = get_best_device()
        else:
            self.device = device

        self._model: BeautyModel | None = None
        self._pretrained_loaded = False

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _get_model(self) -> BeautyModel:
        """Lazy initialization of beauty model."""
        if self._model is None:
            self._model = BeautyModel()
            self._model.to(self.device) # Move to device before loading weights usually better

            # Try to load pretrained weights
            model_path = get_model_path()

            if not model_path.exists():
                logger.info("Downloading beauty prediction model...")
                if download_from_gdrive(GDRIVE_FILE_ID, model_path):
                    logger.info("Beauty model downloaded successfully.")
                else:
                    logger.warning(
                        "Could not download pretrained beauty model. "
                        "Using ImageNet-pretrained backbone only (less accurate)."
                    )

            if model_path.exists():
                try:
                    # Load the state dict
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                    # Handle Checkpoint Dictionaries
                    if isinstance(checkpoint, dict):
                         if "state_dict" in checkpoint:
                             state_dict = checkpoint["state_dict"]
                         elif "model" in checkpoint:
                             state_dict = checkpoint["model"]
                         else:
                             state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # --- Robust Weight Remapping ---
                    new_state_dict = {}
                    import re
                    
                    for k, v in state_dict.items():
                        new_k = k
                        
                        # 1. Strip 'module.' prefix (DataParallel)
                        if new_k.startswith("module."):
                            new_k = new_k[7:]
                        
                        # 2. Map 'fullyconnected' -> 'backbone.fc' (or 'fc' if applied to backbone directly)
                        # We are loading into 'self._model.backbone' mostly, but let's see.
                        # The keys likely correspond to the ResNet itself. 
                        # If we load into self._model.backbone, keys should look like 'conv1...', 'layer1...'
                        # If we load into self._model, keys should look like 'backbone.conv1...'
                        
                        # The loaded weights seem to have keys like 'layer1...' (after stripping module)
                        # So they match the backbone structure directly.
                        
                        if "fullyconnected" in new_k:
                            new_k = new_k.replace("fullyconnected", "fc")
                            
                        # 3. Strip internal 'groupX' which implies ResNeXt-like naming or older Torch conversion
                        # Example: layer4.1.group1.bn2.bias -> layer4.1.bn2.bias
                        # Example: group1.conv1.weight -> conv1.weight
                        # Regex to remove group\d+ followed by dot or preceded by dot
                        new_k = re.sub(r'^group\d+\.', '', new_k)
                        new_k = re.sub(r'\.group\d+', '', new_k)
                        
                        # 4. Map 'backbone.' prefix if needed? 
                        # Since our BeautyModel HAS a .backbone attribute, if we load state_dict into BeautyModel,
                        # we need keys to start with 'backbone.'.
                        # The loaded keys (e.g. 'layer4...') belong to resnet.
                        # So we prefix them with 'backbone.'
                        if not new_k.startswith("backbone."):
                            new_k = f"backbone.{new_k}"
                            
                        new_state_dict[new_k] = v

                    # Load
                    keys = self._model.load_state_dict(new_state_dict, strict=False)
                    
                    missing = [k for k in keys.missing_keys if "num_batches_tracked" not in k]
                    unexpected = keys.unexpected_keys
                    
                    if len(missing) > 0 or len(unexpected) > 0:
                        logger.warning(f"Weight loading loose match. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                        if len(missing) < 20: # Log only if few
                            logger.debug(f"Missing keys: {missing}")
                    
                    self._pretrained_loaded = True
                    logger.info("Loaded pretrained beauty model weights.")

                except Exception as e:
                    logger.warning(f"Failed to load beauty model: {e}")
                    self._pretrained_loaded = False

            self._model.eval()

        return self._model

    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for the model.

        Args:
            face_image: RGB face crop as numpy array

        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure RGB format
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)

        # Apply transforms
        tensor = self.transform(face_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor

    def predict(self, face_image: np.ndarray) -> float:
        """
        Predict beauty score for a face image.

        Args:
            face_image: RGB face crop as numpy array

        Returns:
            Beauty score (typically 1-5 range, but may vary without pretrained weights)
        """
        if face_image.size == 0:
            return 0.0

        try:
            model = self._get_model()
            tensor = self.preprocess(face_image).to(self.device)

            with torch.no_grad():
                score = model(tensor)
                # Clamp to reasonable range and return
                score_value = score.item()

                # If we don't have pretrained weights, normalize the output
                if not self._pretrained_loaded:
                    # Use sigmoid to get 0-1 range, then scale to 1-5
                    score_value = torch.sigmoid(score).item() * 4 + 1
                else:
                    # Clamp to 1-5 range
                    score_value = max(1.0, min(5.0, score_value))

                return score_value

        except Exception as e:
            logger.warning(f"Beauty prediction failed: {e}")
            return 0.0
