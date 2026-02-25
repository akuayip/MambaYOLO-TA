# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

__version__ = "8.1.0"

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "YOLO",
    "ASSETS",
    "SETTINGS",
    "checks",
    "download",
)
