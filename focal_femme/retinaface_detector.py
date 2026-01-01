"""RetinaFace-based face detection module with ResNet50 backbone."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import from copied RetinaFace code
from .retinaface.models import RetinaFace
from .retinaface.config import get_config
from .retinaface.layers.functions.prior_box import PriorBox
from .retinaface.utils.box_utils import decode, decode_landmarks, nms

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """RetinaFace face detector with ResNet50 backbone."""

    def __init__(self, device: torch.device, confidence_threshold: float = 0.8):
        """
        Initialize RetinaFace detector.

        Args:
            device: PyTorch device (cuda/xpu/mps/cpu)
            confidence_threshold: Minimum confidence for face detection (default: 0.8)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        self.pre_nms_topk = 5000
        self.post_nms_topk = 750

        # RGB mean for normalization (BGR format: 104, 117, 123)
        self.rgb_mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)

        self._model = None
        self._cfg = None

    def _get_config(self):
        """Get RetinaFace configuration for ResNet50."""
        if self._cfg is None:
            self._cfg = get_config('resnet50')
            if self._cfg is None:
                raise ValueError("Could not load ResNet50 config")
        return self._cfg

    def _get_model(self) -> RetinaFace:
        """Lazy initialization of RetinaFace model."""
        if self._model is None:
            cfg = self._get_config()

            # Load model architecture
            self._model = RetinaFace(cfg=cfg)

            # Download and load pretrained weights
            model_path = self._get_model_path()
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)

            self._model.to(self.device)
            self._model.eval()
            logger.debug(f"RetinaFace ResNet50 loaded on {self.device}")

        return self._model

    def _get_model_path(self) -> Path:
        """Get path to model weights, downloading if necessary."""
        cache_dir = Path.home() / ".cache" / "focal-femme"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / "retinaface_resnet50.pth"

        if not model_path.exists():
            logger.info("Downloading RetinaFace ResNet50 model...")
            # Download from GitHub releases
            url = "https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth"
            self._download_file(url, model_path)
            logger.info("RetinaFace model downloaded successfully.")

        return model_path

    def _download_file(self, url: str, destination: Path):
        """Download file from URL."""
        import requests
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of face detections, each containing:
            - bbox: (x1, y1, x2, y2) in original image coordinates
            - confidence: Detection confidence score
            - landmarks: 5-point facial landmarks as numpy array (5, 2)
        """
        if image.size == 0:
            return []

        try:
            model = self._get_model()
            cfg = self._get_config()

            img_height, img_width = image.shape[:2]

            # Preprocessing
            img_tensor = self._preprocess(image)
            img_tensor = img_tensor.to(self.device)

            # Inference
            with torch.no_grad():
                loc, conf, landmarks = model(img_tensor)

            # Remove batch dimension
            loc = loc.squeeze(0)
            conf = conf.squeeze(0)
            landmarks = landmarks.squeeze(0)

            # Generate anchor boxes (priors)
            priorbox = PriorBox(cfg, image_size=(img_height, img_width))
            priors = priorbox.generate_anchors().to(self.device)

            # Decode boxes and landmarks
            boxes = decode(loc, priors, cfg['variance'])
            landmarks_decoded = decode_landmarks(landmarks, priors, cfg['variance'])

            # Scale to original image dimensions
            bbox_scale = torch.tensor([img_width, img_height] * 2, device=self.device)
            boxes = (boxes * bbox_scale).cpu().numpy()

            landmark_scale = torch.tensor([img_width, img_height] * 5, device=self.device)
            landmarks_decoded = (landmarks_decoded * landmark_scale).cpu().numpy()

            # Get scores (probability of face class)
            scores = conf.cpu().numpy()[:, 1]

            # Post-processing: filter, sort, NMS
            detections = self._postprocess(boxes, landmarks_decoded, scores)

            return detections

        except Exception as e:
            logger.warning(f"RetinaFace detection failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for RetinaFace.

        Args:
            image: RGB image as numpy array

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert RGB to BGR (RetinaFace expects BGR)
        img_bgr = image[:, :, ::-1].copy()

        # Convert to float32
        img_float = img_bgr.astype(np.float32)

        # Subtract mean (BGR mean: 104, 117, 123)
        img_float -= self.rgb_mean

        # Transpose to CHW and add batch dimension (HWC -> CHW -> 1CHW)
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)

        return img_tensor

    def _postprocess(
        self,
        boxes: np.ndarray,
        landmarks: np.ndarray,
        scores: np.ndarray
    ) -> list[dict]:
        """
        Post-process RetinaFace outputs: filter, sort, NMS.

        Args:
            boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2]
            landmarks: Facial landmarks (N, 10) as [x1, y1, x2, y2, ..., x5, y5]
            scores: Confidence scores (N,)

        Returns:
            List of dicts with bbox, confidence, landmarks
        """
        # Filter by confidence threshold
        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        if len(boxes) == 0:
            return []

        # Sort by scores (descending)
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_threshold)

        detections = detections[keep]
        landmarks = landmarks[keep]

        # Keep top-k detections
        detections = detections[:self.post_nms_topk]
        landmarks = landmarks[:self.post_nms_topk]

        # Format output as list of dicts
        result = []
        for i in range(len(detections)):
            bbox = detections[i, :4]  # [x1, y1, x2, y2]
            confidence = float(detections[i, 4])
            landmark_points = landmarks[i].reshape(5, 2)  # (5, 2)

            result.append({
                'bbox': bbox,
                'confidence': confidence,
                'landmarks': landmark_points
            })

        return result
