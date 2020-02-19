"""Contains utility functions for Yolo v3 model."""

import numpy as np
from seaborn import color_palette
import cv2



def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def clasifier(boxes_dicts, class_names):
    """Draws detected boxes in a video frame.
    Args:
        boxes_dicts: A class-to-boxes dictionary.
        class_names: A class names list.
    Returns:
        None.
    """
    boxes_dict = boxes_dicts[0]
    classify = {}

    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        
        
        if np.size(boxes) != 0:
            for box in boxes:
                confidence = box[4]
                classify[class_names[cls]] = confidence * 100

    return classify