import numpy as np
from numpy.typing import NDArray


def class_aware_non_maximum_suppression(
    scores: NDArray[np.float32],
    labels: NDArray[np.float32],
    boxes: NDArray[np.float32],
    threshold: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    result_boxes = []
    result_scores = []
    result_labels = []

    labels_unique = np.unique(labels)
    for label in labels_unique:
        boxes_label = np.array(
            [box for i, box in enumerate(boxes) if labels[i] == label]
        )
        scores_label = [score for i, score in enumerate(scores) if labels[i] == label]
        labels_label = [label] * len(boxes)
        result_scores_label, result_labels_label, result_boxes_label = (
            non_maximum_suppression(scores_label, labels_label, boxes_label, threshold)
        )
        result_boxes += result_boxes_label
        result_scores += result_scores_label
        result_labels += result_labels_label
    return np.array(result_scores), np.array(result_labels), np.array(result_boxes)


def non_maximum_suppression(
    scores: NDArray[np.float32],
    labels: NDArray[np.float32],
    boxes: NDArray[np.float32],
    threshold: float,
) -> tuple[list[float], list[float], list[float]]:
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    areas = (end_y - start_y) * (end_x - start_x)

    score_indices = np.argsort(scores)
    result_boxes = []
    result_scores = []
    result_labels = []

    while score_indices.size > 0:
        index = score_indices[-1]
        result_boxes.append(boxes[index])
        result_scores.append(scores[index])
        result_labels.append(labels[index])

        x1 = np.maximum(start_x[index], start_x[score_indices[:-1]])
        x2 = np.minimum(end_x[index], end_x[score_indices[:-1]])
        y1 = np.maximum(start_y[index], start_y[score_indices[:-1]])
        y2 = np.minimum(end_y[index], end_y[score_indices[:-1]])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)

        intersection = w * h
        iou = intersection / (areas[index] + areas[score_indices[:-1]] - intersection)

        left = np.where(iou < threshold)
        score_indices = score_indices[left]

    return result_scores, result_labels, result_boxes
