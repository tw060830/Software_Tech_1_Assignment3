from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImageRecord:
    """The class name that represents a single indexed macroinvertebrate image record."""

    file_path: Path
    label: str
    width: int
    height: int
    channels: int