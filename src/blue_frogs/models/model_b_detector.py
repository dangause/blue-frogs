"""Model B Stage 1: Frog detection and cropping using YOLOv8 or MegaDetector."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrogDetector:
    """Detect and crop frogs from images using a pretrained detector."""

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            if self._model_path:
                self._model = YOLO(self._model_path)
            else:
                # Default: YOLOv8 pretrained on COCO (has 'frog' class)
                self._model = YOLO("yolov8m.pt")

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Detect frogs in an image. Returns list of bounding boxes."""
        self._load_model()
        results = self._model(image_rgb, verbose=False, conf=self.conf_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                # COCO class 'frog' = id 30
                if cls_name == "frog" or cls_id == 30:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                    })
        return detections

    def crop_frog(
        self, image_rgb: np.ndarray, padding_fraction: float = 0.1
    ) -> np.ndarray | None:
        """Detect and crop the most confident frog from the image.

        Returns the cropped image or None if no frog detected.
        Falls back to center crop if detection fails.
        """
        detections = self.detect(image_rgb)
        if not detections:
            # Fallback: center crop (80% of image)
            h, w = image_rgb.shape[:2]
            margin_h, margin_w = int(h * 0.1), int(w * 0.1)
            return image_rgb[margin_h:h - margin_h, margin_w:w - margin_w]

        # Take highest confidence detection
        best = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]

        # Add padding
        h, w = image_rgb.shape[:2]
        pad_h = int((y2 - y1) * padding_fraction)
        pad_w = int((x2 - x1) * padding_fraction)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return image_rgb[y1:y2, x1:x2]
