# Ultralytics YOLO 🚀, AGPL-3.0 license
# Refactored for Mamba-YOLO Object Detection only

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """Mamba-YOLO object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model."""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
