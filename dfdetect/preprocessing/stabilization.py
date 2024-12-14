import os
import cv2
import numpy as np
from typing import List, Optional
from dfdetect.preprocessing.face_detection import BBox
from collections import defaultdict

try:
    from vidstab import VidStab, layer_overlay
    from vidstab.vidstab_utils import build_transformation_matrix
except ImportError:
    pass


def stabilize(frames, stabilizing_window=30):
    stabilizer = VidStab()
    if len(frames) < stabilizing_window:
        print("Not enough frames to stabilize with this window")
        return

    for i, frame in enumerate(frames):
        frame_stabilized = stabilizer.stabilize_frame(
            input_frame=frame,
            smoothing_window=stabilizing_window,
            border_type="reflect",
        )
        if i < stabilizing_window:
            continue
        yield frame_stabilized

    for i in range(stabilizing_window - 1):
        # empty frames buffer
        frame_stabilized = stabilizer.stabilize_frame(
            input_frame=None, smoothing_window=stabilizing_window, border_type="reflect"
        )
        yield frame_stabilized


def crop_and_stabilize(
    frames: List[np.ndarray], bboxes: List[Optional[BBox]], stabilizing_window=30
):
    """Use vidstab to stabilize frames"""
    if len(frames) < stabilizing_window:
        if len(frames) > 10:
            stabilizing_window = 10
        else:
            print("Not enough frames to stabilize")
            return

    assert len(frames) == len(bboxes)

    stabilizer = VidStab()

    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is None:
            continue

        cropped_frame = frame[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]

        # # Attempt at adding back the background the hard way
        # def layer_blending(foreground, background):
        #     transform = build_transformation_matrix(stabilizer.transforms[-1])

        #     h, w = frame.shape[:2]
        #     transformed_frame_image = cv2.warpAffine(frame, transform, (w, h), borderMode=cv2.BORDER_REFLECT)

        #     zero_frame = np.zeros_like(frame)
        #     zero_frame[bbox.y0, bbox.x0, 0] = 1
        #     transformed_zero_image = cv2.warpAffine(zero_frame, transform, (w, h), borderMode=cv2.BORDER_REFLECT)

        #     ny, nx = np.nonzero(transformed_zero_image[:, :, 0])

        #     if not len(ny) or not len(nx):
        #         return layer_overlay(foreground, background)

        #     MPy = ny[0]
        #     MPx = nx[0]

        #     # transform = np.vstack((transform, [0, 0, 1]))

        #     # bbox_arr = np.array(
        #     #     [[[bbox.x0, bbox.y0]], [[bbox.x1, bbox.y1]], [[bbox.x0, bbox.y1]], [[bbox.x1, bbox.y1]]],
        #     #     dtype=np.float32,
        #     # )
        #     # bbox_transformed = np.squeeze(cv2.perspectiveTransform(bbox_arr, transform).astype(int))
        #     # ul = bbox_transformed[0]
        #     # dr = bbox_transformed[1]

        #     overlaid = foreground.copy()
        #     negative_space = np.where(foreground[:, :, 3] == 0)
        #     yi, xi = negative_space

        #     # xi + ul[0]
        #     # yi + ul[1]

        #     # result = cv2.matchTemplate(foreground[:, :, :3], transformed_frame_image, #cv2.TM_SQDIFF_NORMED)
        #     # mn, _, (MPx, MPy), _ = cv2.minMaxLoc(result)

        #     overlaid[yi, xi, :3] = transformed_frame_image[yi + MPy, xi + MPx, :]
        #     overlaid[:, :, 3] = 255

        #     return overlaid

        frame_stabilized = stabilizer.stabilize_frame(
            input_frame=cropped_frame,
            smoothing_window=stabilizing_window,
            border_type="reflect",
            # layer_func=layer_blending,
        )
        if i < stabilizing_window:
            continue

        yield frame_stabilized

    for i in range(stabilizing_window - 1):
        # empty frames buffer
        frame_stabilized = stabilizer.stabilize_frame(
            input_frame=None,
            smoothing_window=stabilizing_window,
            # layer_func=layer_blending,
            border_type="reflect",
        )
        if frame_stabilized is None:
            break

        yield frame_stabilized


def get_opencv_features(frame: np.ndarray) -> np.ndarray:
    features = np.empty(0)
    quality_level = 0.01
    nb_iters = 0

    while np.size(features) < 3:
        features = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=200,
            qualityLevel=quality_level,
            minDistance=30,
            blockSize=3,
        )
        quality_level /= 10
        nb_iters += 1
        if nb_iters > 10:
            print("Error trying to find good features")
            break

    return features


def crop_and_stabilize2(
    frames: List[np.ndarray],
    bboxes: List[Optional[BBox]],
    stabilizing_window=30,
    padding=0.15,
):
    """Use opencv feature to stabilize frames, and then crop on bbox, stabilization is used using pixels inside the bboxes"""
    assert len(frames) == len(bboxes)

    moving_avg_dict = defaultdict(list)

    nb_frames = len(frames)
    for i in range(1, nb_frames):
        prev_frame = frames[i - 1]
        prev_bbox = bboxes[i - 1]
        curr_frame = frames[i]
        curr_bbox = bboxes[i]
        if prev_bbox is None or curr_bbox is None:
            continue
        h, w, _ = prev_frame.shape

        cropped_prev_frame = prev_frame[
            prev_bbox.y0 : prev_bbox.y1, prev_bbox.x0 : prev_bbox.x1
        ]
        cropped_curr_frame = curr_frame[
            curr_bbox.y0 : curr_bbox.y1, curr_bbox.x0 : curr_bbox.x1
        ]

        cropped_prev_frame = cv2.cvtColor(cropped_prev_frame, cv2.COLOR_RGB2GRAY)
        cropped_curr_frame_bw = cv2.cvtColor(cropped_curr_frame, cv2.COLOR_RGB2GRAY)

        feat_prev_frame = get_opencv_features(cropped_prev_frame)
        feat_curr_frame, status, err = cv2.calcOpticalFlowPyrLK(
            cropped_prev_frame, cropped_curr_frame_bw, feat_prev_frame, None
        )
        idx = np.where(status == 1)[:5]
        feat_prev_frame = feat_prev_frame[idx]
        feat_curr_frame = feat_curr_frame[idx]
        if np.size(feat_prev_frame) == 0:
            print("Insufficient number of features found")
            continue
        m, _ = cv2.estimateAffinePartial2D(feat_prev_frame, feat_curr_frame)
        if m is None:
            print("Failed to estimate affine transform")
            continue

        moving_avg_dict["dx"].append(m[0, 2])
        moving_avg_dict["dy"].append(m[1, 2])
        moving_avg_dict["da"].append(np.arctan2(m[1, 0], m[0, 0]))

        dxs = moving_avg_dict["dx"][-stabilizing_window:]
        dys = moving_avg_dict["dy"][-stabilizing_window:]
        das = moving_avg_dict["da"][-stabilizing_window:]

        traj = np.cumsum((dxs, dys, das), axis=1)
        smooth_traj = np.mean(traj, axis=1, keepdims=True)
        err = (smooth_traj - traj)[:, -1]

        dx = dxs[-1] + err[0]
        dy = dys[-1] + err[1]
        da = das[-1] + err[2]

        m = np.array([[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]])
        # m = np.empty((2, 3), np.float32)
        # m[0, 0] = np.cos(da)
        # m[0, 1] = -np.sin(da)
        # m[1, 0] = np.sin(da)
        # m[1, 1] = np.cos(da)
        # m[0, 2] = dx
        # m[1, 2] = dy

        m_padded = np.vstack((m, (0, 0, 1)))
        rect_pts = np.array(
            [
                [[curr_bbox.x0, curr_bbox.y0]],
                [[curr_bbox.x1, curr_bbox.y0]],
                [[curr_bbox.x0, curr_bbox.y1]],
                [[curr_bbox.x1, curr_bbox.y1]],
            ],
            dtype=np.float32,
        )
        rect_bbox = np.squeeze(cv2.perspectiveTransform(rect_pts, m_padded))
        x0 = int((rect_bbox[0, 0] + rect_bbox[2, 0]) * 0.5) % w
        x1 = int((rect_bbox[1, 0] + rect_bbox[3, 0]) * 0.5) % w
        y0 = int((rect_bbox[0, 1] + rect_bbox[1, 1]) * 0.5) % h
        y1 = int((rect_bbox[2, 1] + rect_bbox[3, 1]) * 0.5) % h

        center_x = (x1 - x0) // 2 + x0
        center_y = (y1 - y0) // 2 + y0
        cropped_w = int(curr_bbox.width() * (1 + padding))
        cropped_h = int(curr_bbox.height() * (1 + padding))

        y0 = max((center_y - cropped_h // 2), 0)
        y1 = center_y + cropped_h // 2
        x0 = max(center_x - cropped_w // 2, 0)
        x1 = center_x + cropped_w // 2

        frame_stabilized = cv2.warpAffine(curr_frame, m, (w, h))
        frame_stabilized = frame_stabilized[y0:y1, x0:x1]

        yield frame_stabilized
