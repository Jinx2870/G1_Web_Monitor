from pathlib import Path

import cv2
import yaml


DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
    },
    "camera": {
        "source": "realsense",
        "webcam_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "enable_rgb": True,
        "enable_depth": True,
        "align_depth": True,
        "depth_colormap": "JET",
        "jpeg_quality": 85,
    },
}


COLORMAP_MAP = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "TURBO": cv2.COLORMAP_TURBO,
}


def _merge_dict(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path="config.yaml"):
    path = Path(config_path)
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_CONFIG

    with path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    return _merge_dict(DEFAULT_CONFIG, user_config)


def get_colormap(colormap_name):
    return COLORMAP_MAP.get(str(colormap_name).upper(), cv2.COLORMAP_JET)
