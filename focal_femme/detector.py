"""Face detection and gender classification module using facenet-pytorch."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from .utils import FaceData, bbox_area, bbox_center_distance, load_image
from .gender_model import GenderClassifier
from .beauty_model import BeautyScorer

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Intermediate structure for a detected face before filtering."""

    bbox: tuple[int, int, int, int]  # (top, right, bottom, left)
    embedding: np.ndarray
    area: int
    center_distance: float
    is_female: bool
    female_confidence: float
    face_image: np.ndarray  # Cropped face for gender classification
    beauty_score: float = 0.0


class FaceDetector:
    """Handles face detection, gender classification, and primary face selection."""

    def __init__(
        self,
        female_threshold: float = 0.5,
        device: str | None = None,
    ):
        """
        Initialize the face detector.

        Args:
            female_threshold: Minimum confidence for female classification (0-1)
            device: Torch device ('cuda', 'cpu', or None for auto)
        """
        self.female_threshold = female_threshold

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._mtcnn: MTCNN | None = None
        self._resnet: InceptionResnetV1 | None = None
        self._gender_model: GenderClassifier | None = None
        self._beauty_model: BeautyScorer | None = None

    def _get_mtcnn(self) -> MTCNN:
        """Lazy initialization of MTCNN face detector."""
        if self._mtcnn is None:
            self._mtcnn = MTCNN(
                image_size=160,
                margin=20,
                keep_all=True,
                post_process=True,  # Normalize to [-1, 1] for InceptionResnetV1
                device=self.device,
            )
        return self._mtcnn

    def _get_resnet(self) -> InceptionResnetV1:
        """Lazy initialization of face embedding model."""
        if self._resnet is None:
            self._resnet = InceptionResnetV1(
                pretrained='vggface2',
                device=self.device,
            ).eval()
        return self._resnet

    def _get_gender_model(self) -> GenderClassifier:
        """Lazy initialization of gender classifier."""
        if self._gender_model is None:
            self._gender_model = GenderClassifier()
        return self._gender_model

    def _get_beauty_model(self) -> BeautyScorer:
        """Lazy initialization of beauty scorer."""
        if self._beauty_model is None:
            self._beauty_model = BeautyScorer(device=self.device)
        return self._beauty_model

    def detect_and_analyze_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in an image, extract embeddings and classify gender.

        Args:
            image: RGB image as numpy array

        Returns:
            List of DetectedFace objects
        """
        height, width = image.shape[:2]
        detected_faces: list[DetectedFace] = []

        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Detect faces
            mtcnn = self._get_mtcnn()
            boxes, probs = mtcnn.detect(pil_image)

            if boxes is None:
                return []

            # Get face crops for embedding extraction
            faces_tensor = mtcnn(pil_image)

            if faces_tensor is None:
                return []

            # Get embeddings (L2 normalized for better clustering)
            resnet = self._get_resnet()
            with torch.no_grad():
                embeddings = resnet(faces_tensor.to(self.device))
                # L2 normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.cpu().numpy()

            # Get gender classifier and beauty scorer
            gender_model = self._get_gender_model()
            beauty_model = self._get_beauty_model()

            # Process each detected face
            for i, (box, prob, embedding) in enumerate(zip(boxes, probs, embeddings)):
                if prob < 0.75:  # Skip low-confidence detections
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Ensure bounds are within image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                # Convert to (top, right, bottom, left) format
                bbox = (y1, x2, y2, x1)
                area = bbox_area(bbox)
                center_dist = bbox_center_distance(bbox, width, height)

                # Extract face crop for gender classification
                face_crop = image[y1:y2, x1:x2]

                # Classify gender
                is_female, female_conf = gender_model.predict(face_crop)
                is_female = female_conf >= self.female_threshold

                # Predict beauty score
                beauty_score = beauty_model.predict(face_crop)

                detected_faces.append(DetectedFace(
                    bbox=bbox,
                    embedding=embedding,
                    area=area,
                    center_distance=center_dist,
                    is_female=is_female,
                    female_confidence=female_conf,
                    face_image=face_crop,
                    beauty_score=beauty_score,
                ))

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

        return detected_faces

    def select_primary_face(
        self,
        faces: list[DetectedFace],
    ) -> DetectedFace | None:
        """
        Select the primary female face from detected faces.

        Strategy:
        1. Filter for female faces
        2. Select the largest by bounding box area
        3. Use centrality as tiebreaker

        Args:
            faces: List of detected faces

        Returns:
            Primary face or None
        """
        if not faces:
            return None

        # Filter for female faces
        female_faces = [f for f in faces if f.is_female]

        if not female_faces:
            return None

        # Sort by area (descending), then by center distance (ascending)
        female_faces.sort(key=lambda f: (-f.area, f.center_distance))

        return female_faces[0]

    def process_image(self, image_path: Path, verbose: bool = False) -> FaceData | None:
        """
        Process a single image and extract primary female face data.

        Args:
            image_path: Path to the image file
            verbose: If True, log detailed info about detected faces

        Returns:
            FaceData for the primary female face, or None if not found
        """
        try:
            image = load_image(image_path)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

        faces = self.detect_and_analyze_faces(image)

        if not faces:
            if verbose:
                logger.warning(f"SKIPPED {image_path.name}: No faces detected")
            return None

        if verbose:
            # Log details about all detected faces
            logger.warning(f"IMAGE {image_path.name}: {len(faces)} face(s) detected")
            for i, face in enumerate(faces):
                gender_str = "FEMALE" if face.is_female else "MALE"
                logger.warning(
                    f"  Face {i+1}: {gender_str} (conf={face.female_confidence:.2f}), "
                    f"area={face.area}px, beauty={face.beauty_score:.2f}"
                )

        primary_face = self.select_primary_face(faces)

        # Fallback: if no female detected, use the most female-like face
        if primary_face is None and len(faces) >= 1:
            # Sort by female confidence (descending), then by area (descending)
            faces_by_confidence = sorted(faces, key=lambda f: (-f.female_confidence, -f.area))
            primary_face = faces_by_confidence[0]
            if verbose:
                logger.warning(
                    f"  -> Fallback: using face with highest female confidence "
                    f"({primary_face.female_confidence:.2f})"
                )

        if primary_face is None:
            if verbose:
                logger.warning(f"  -> No face selected")
            return None

        if verbose:
            logger.warning(f"  -> Selected female face with area={primary_face.area}px")

        return FaceData(
            file_path=image_path,
            embedding=primary_face.embedding,
            bbox=primary_face.bbox,
            is_female=True,
            confidence=primary_face.female_confidence,
            cluster_id=None,
            beauty_score=primary_face.beauty_score,
        )

    def process_images(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int, int], None] | None = None,
        verbose: bool = False,
    ) -> dict[str, FaceData]:
        """
        Process multiple images and extract primary female face data.

        Args:
            image_paths: List of image file paths
            progress_callback: Optional callback(current, total) for progress
            verbose: If True, log detailed info about each image

        Returns:
            Dictionary mapping file path strings to FaceData
        """
        results: dict[str, FaceData] = {}
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            face_data = self.process_image(image_path, verbose=verbose)

            if face_data is not None:
                results[str(image_path)] = face_data

            if progress_callback:
                progress_callback(idx + 1, total)

        return results
