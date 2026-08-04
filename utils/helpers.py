# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
import os
import platform
import numpy as np

from utils.config import (
    USB_CAMERA_WINDOWS,
    USB_CAMERA_LINUX,
)
CLASS_COLORS = {
    0: (0, 255, 0),      # hijau
    1: (255, 0, 0),      # biru (format OpenCV: BGR)
    2: (0, 0, 255),      # merah
    3: (0, 255, 255),    # kuning
    4: (255, 0, 255),    # magenta
}

def get_usb_camera_index():
    if platform.system() == "Windows":
        return USB_CAMERA_WINDOWS
    return USB_CAMERA_LINUX


def get_class_color(cls_id):
    """
    Return display color for a class.

    - Known classes use predefined colors.
    - Unknown classes receive a deterministic pseudo-random color.
    """
    if cls_id in CLASS_COLORS:
        return CLASS_COLORS[cls_id]

    rng = np.random.default_rng(seed=cls_id)
    return tuple(int(c) for c in rng.integers(0, 256, size=3))

def load_yolo_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.split())
            boxes.append((int(c), (x, y, w, h)))
    return boxes

def build_rtsp_url(ip, username, password, port=554):
    return (
        f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/101"
    )