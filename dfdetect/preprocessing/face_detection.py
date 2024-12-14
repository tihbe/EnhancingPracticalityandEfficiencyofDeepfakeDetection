import torch
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy

# For Blazeface
try:
    import mediapipe as mp
except ModuleNotFoundError:
    pass


def waterfall(frame):
    """Try mutliple face detection methods (from fastest to slowest) to try and detect faces."""
    (bboxes,) = detect_face_blazeface([frame])
    if len(bboxes) == 0:
        (bboxes,) = detect_face_mtcnn([frame])
    if len(bboxes) == 0:
        (bboxes,) = detect_face_retinaface([frame])
    return bboxes


# TODO:
# The face detection methods could be batched over the frames and run on gpu


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int
    center_landmark: Tuple[int, int]

    def width(self) -> int:
        return self.x1 - self.x0

    def height(self) -> int:
        return self.y1 - self.y0

    def recast(self) -> "BBox":
        """Recast as uint"""
        self.x0 = max(int(self.x0), 0)
        self.y0 = max(int(self.y0), 0)
        self.x1 = max(int(self.x1), 0)
        self.y1 = max(int(self.y1), 0)
        self.center_landmark = max(int(self.center_landmark[0]), 0), max(
            int(self.center_landmark[1]), 0
        )
        return self


_MTCNN = None


def detect_face_mtcnn(
    frames: List[np.ndarray], prob_thresh: float = 0.7
) -> List[List[BBox]]:
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN

        _MTCNN = MTCNN(224, margin=30, keep_all=True, post_process=True)

    bboxes = []
    for frame in tqdm(frames, desc="Detecting faces", disable=len(frames) == 1):
        batch_boxes, batch_probs, batch_landmarks = _MTCNN.detect(frame, landmarks=True)

        frame_bboxes = []
        if batch_boxes is not None:
            nb_faces = len(batch_boxes)
            for i in range(nb_faces):
                if batch_probs[i] < prob_thresh:
                    continue

                face_landmarks = batch_landmarks[i]
                dist_to_center = (
                    face_landmarks - face_landmarks.mean(axis=0, keepdims=True)
                ) ** 2
                center_landmark = np.argmin(dist_to_center.sum(axis=1))

                bbox = BBox(*batch_boxes[i], face_landmarks[center_landmark])
                bbox.recast()
                if bbox.width() * bbox.height() != 0:
                    frame_bboxes.append(bbox)

        bboxes.append(frame_bboxes)

    return bboxes


def detect_face_blazeface(
    frames: List[np.ndarray], prob_thresh: float = 0.7
) -> List[List[BBox]]:
    mp_face_detection = mp.solutions.face_detection
    bboxes = []

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=prob_thresh
    ) as face_detection:
        for i, frame in enumerate(frames):
            frame.flags.writeable = False
            results = face_detection.process(frame)

            height, width, _ = frame.shape
            frame_bboxes = []

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    nose = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
                    )

                    c = lambda pos, size: int(pos * size)

                    bbox = BBox(
                        c(bbox.xmin, width),
                        c(bbox.ymin, height),
                        c(bbox.width + bbox.xmin, width),
                        c(bbox.height + bbox.ymin, height),
                        (c(nose.x, width), c(nose.y, height)),
                    )
                    bbox.recast()
                    if bbox.width() * bbox.height() != 0:
                        frame_bboxes.append(bbox)

            bboxes.append(frame_bboxes)
    return bboxes


_RETINAFACE = None


def detect_face_retinaface(
    frames: List[np.ndarray], prob_thresh: float = 0.7
) -> List[List[BBox]]:
    global _RETINAFACE
    if _RETINAFACE is None:
        from retinaface.pre_trained_models import get_model

        _RETINAFACE = get_model("resnet50_2020-07-20", max_size=2048)
        _RETINAFACE.eval()
        if torch.cuda.is_available():
            _RETINAFACE.device = torch.device("cuda")
            _RETINAFACE.model = _RETINAFACE.model.to(_RETINAFACE.device)

    bboxes = []

    with torch.no_grad():
        for frame in tqdm(frames, desc="Detecting faces", disable=len(frames) == 1):
            frame.flags.writeable = False
            annotations = _RETINAFACE.predict_jsons(
                frame, confidence_threshold=prob_thresh
            )

            frame_bboxes = []
            for detection in annotations:
                if "landmarks" not in detection and "bbox" not in detection:
                    continue
                if len(detection["landmarks"]) == 0:
                    continue
                if len(np.shape(detection["landmarks"])) == 1:
                    center_x, center_y = detection["landmarks"]
                else:
                    center_x, center_y = np.mean(detection["landmarks"], axis=0)

                bbox = BBox(*detection["bbox"], (center_x, center_y))
                bbox.recast()
                if bbox.width() * bbox.height() != 0:
                    frame_bboxes.append(bbox)

            bboxes.append(frame_bboxes)

    return bboxes


def face_tracking(
    bboxes: List[List[Optional[BBox]]],
) -> Dict[int, List[Optional[BBox]]]:
    """Organize the bboxes by face_id by tracking the movement of the center landmark"""
    tracked_faces = defaultdict(list)
    face_ids = {}
    for i, frame_bboxes in enumerate(bboxes):
        processed_faces = []
        for face_bbox in frame_bboxes:
            if face_bbox is None:
                continue
            center_x, center_y = face_bbox.center_landmark

            if len(face_ids) == 0:  # First face
                face_id = 0
            else:
                last_centers = np.array(list(face_ids.values()))
                distances_to_last_centers = (center_x - last_centers[:, 0]) ** 2 + (
                    center_y - last_centers[:, 1]
                ) ** 2
                candidates = distances_to_last_centers < 0.4 * (
                    face_bbox.width() ** 2 + face_bbox.height() ** 2
                )

                nb_matchs = np.count_nonzero(candidates)
                if nb_matchs == 0:  # New face id
                    face_id = len(face_ids)
                elif nb_matchs == 1:  # One face match
                    face_id = np.flatnonzero(candidates)[0]
                else:  # Multiple matches, find closest
                    face_id = np.argmin(distances_to_last_centers)

            if len(tracked_faces[face_id]) <= i:
                face_ids[face_id] = (center_x, center_y)
                tracked_faces[face_id].append(face_bbox)
            else:  # If there is more than one bbox for the same face at the same frame
                dimension_tracking = True
                if (
                    len(tracked_faces[face_id]) > 2
                ):  # Try to minimize center landmark movement
                    dimension_tracking = False
                    last_bbox, before_last_bbox = None, None
                    for tf in reversed(tracked_faces[face_id]):
                        if tf is not None:
                            if last_bbox is None:
                                last_bbox = tf
                            else:
                                before_last_bbox = tf
                                break
                    if before_last_bbox is not None:
                        cx, cy = before_last_bbox.center_landmark

                        if (last_bbox.center_landmark[0] - cx) ** 2 + (
                            last_bbox.center_landmark[1] - cy
                        ) ** 2 > (face_bbox.center_landmark[0] - cx) ** 2 + (
                            face_bbox.center_landmark[1] - cy
                        ) ** 2:
                            tracked_faces[face_id][-1] = face_bbox
                    else:
                        dimension_tracking = True
                if dimension_tracking:  # Or else, just pick based on dimensions
                    last_bbox = next(
                        tf for tf in reversed(tracked_faces[face_id]) if tf is not None
                    )
                    if (
                        last_bbox.width() * last_bbox.height()
                        < face_bbox.width() * face_bbox.height()
                    ):
                        tracked_faces[face_id][-1] = face_bbox

            processed_faces.append(face_id)

        unprocessed_faces = set(face_ids.keys()) - set(processed_faces)
        for face_id in unprocessed_faces:
            tracked_faces[face_id].append(None)

    # Pre-Pad with Nones
    for face_id in face_ids.keys():
        if len(tracked_faces[face_id]) < len(bboxes):
            nb_missing_frames = len(bboxes) - len(tracked_faces[face_id])
            tracked_faces[face_id] = [None] * nb_missing_frames + tracked_faces[face_id]
    return tracked_faces


def max_size_bbox(bboxes: List[Optional[BBox]]) -> Tuple[int, int]:
    max_width = max(b.width() for b in bboxes if b is not None)
    max_height = max(b.height() for b in bboxes if b is not None)
    return (max_width, max_height)


def smooth_bboxes_centers(bboxes: List[Optional[BBox]], window=10) -> List[BBox]:
    """Either do a moving average on the center landmark, or fill the empty bboxes
    where the bbox are None with a linear transition between bboxes"""
    empty_mask = np.array([bbox is None for bbox in bboxes])
    transitions = np.flatnonzero(empty_mask[:-1] != empty_mask[1:])
    output_bboxes = copy.deepcopy(bboxes)

    # Remove no bboxes
    for begin, end in zip(transitions[:-1] + 1, transitions[1:]):
        if empty_mask[begin]:
            if begin - 1 >= 0 and end + 1 < len(bboxes):
                start_pt = bboxes[begin - 1].center_landmark
                end_pt = bboxes[end + 1].center_landmark

                centers_x = np.linspace(
                    start_pt[0], end_pt[0], end - begin + 3, dtype=int
                )[1:-1]
                centers_y = np.linspace(
                    start_pt[1], end_pt[1], end - begin + 3, dtype=int
                )[1:-1]
                for i, center_landmark in enumerate(zip(centers_x, centers_y)):
                    new_bbox = copy.deepcopy(bboxes[begin - 1])
                    new_bbox.center_landmark = center_landmark
                    output_bboxes[begin + i] = new_bbox

    # Moving average of center landmark
    moving_avg = lambda x: (
        np.convolve(x, np.ones(window) / window, mode="same").astype(int)
        if len(x)
        else []
    )
    empty_mask = np.array([bbox is None for bbox in output_bboxes])
    transitions = np.flatnonzero(empty_mask[:-1] != empty_mask[1:])
    assert (
        len(transitions) <= 2
    )  # Assert that the previous part worked at removing Nones
    begin = transitions[0] + 1 if empty_mask[0] else 0
    end = transitions[-1] if empty_mask[-1] else len(bboxes) - 1

    if (end - begin) > 2 * window:
        valid_bboxes = output_bboxes[begin:end]
        centers_x = [b.center_landmark[0] for b in valid_bboxes]
        centers_x = moving_avg(centers_x)
        centers_y = [b.center_landmark[1] for b in valid_bboxes]
        centers_y = moving_avg(centers_y)
        for i, bbox_id in enumerate(range(begin + window, end - window)):
            output_bboxes[bbox_id].center_landmark = (
                centers_x[i + window],
                centers_y[i + window],
            )

    return output_bboxes


def recenter_bboxes(
    bboxes: List[Optional[BBox]], width, height, frame_width, frame_height
) -> List[BBox]:
    """Center the bboxes to the center landmark with fixed width and height"""
    centered_bboxes = []

    for bbox in bboxes:
        if bbox is None:
            centered_bboxes.append(None)
            continue
        center_x, center_y = bbox.center_landmark
        centered_bbox = BBox(
            max(center_x - width // 2, 0),
            max(center_y - height // 2, 0),
            min(center_x + width // 2, frame_width),
            min(center_y + height // 2, frame_height),
            bbox.center_landmark,
        )
        # Translate if necessary to keep (width, height) constant
        if centered_bbox.width() < width:
            dwidth = width - centered_bbox.width()
            if centered_bbox.x0 > 0:
                dwidth *= -1
            centered_bbox.x0 += dwidth
            centered_bbox.x1 += dwidth
        if centered_bbox.height() < height:
            dheight = height - centered_bbox.height()
            if centered_bbox.y0 > 0:
                dheight *= -1
            centered_bbox.y0 += dheight
            centered_bbox.y1 += dheight

        centered_bboxes.append(centered_bbox)

    return centered_bboxes


# def extract_face_with_margin(frame, center, width, height, margin=80):
#     """Extract face from center position and width+margin x height+margin around it"""
#     bbox = [
#         int(max(center[0] - width / 2 - margin / 2, 0)),
#         int(max(center[1] - height / 2 - margin / 2, 0)),
#         int(min(center[0] + width / 2 + margin / 2, frame.shape[1])),
#         int(min(center[1] + height / 2 + margin / 2, frame.shape[0])),
#     ]

#     return frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]


# def detect_face_mtcnn_full(frames, face_center_window=30, output_image_size=224, prob_thresh=0.9):
#     global _MTCNN
#     if _MTCNN is None:
#         from facenet_pytorch import MTCNN

#         _MTCNN = MTCNN(224, margin=30, keep_all=True, post_process=True)

#     face_frames = defaultdict(list)
#     face_size_queues = {}
#     face_ids = {}
#     for frame in tqdm(frames, desc="Detecting faces"):
#         batch_boxes, batch_probs, batch_landmarks = _MTCNN.detect(frame, landmarks=True)
#         if batch_boxes is None or (nb_faces := len(batch_boxes)) == 0:  # no face detected
#             continue

#         for i in range(nb_faces):
#             if batch_probs[i] < prob_thresh:
#                 continue

#             face_landmarks = batch_landmarks[i]
#             dist_to_center = (face_landmarks - face_landmarks.mean(axis=0, keepdims=True)) ** 2
#             center_landmark = np.argmin(dist_to_center.sum(axis=1))
#             (center_x, center_y) = face_landmarks[center_landmark]

#             bbox = batch_boxes[i]
#             face_width = bbox[2] - bbox[0]
#             face_height = bbox[3] - bbox[1]

#             # Face tracking from center of face
#             for face_id, face_center in face_ids.items():
#                 if (face_center[0] - center_x) ** 2 + (
#                     face_center[1] - center_y
#                 ) ** 2 < face_width**2 + face_height**2:
#                     break
#             else:
#                 face_id = len(face_ids)
#             face_ids[face_id] = (center_x, center_y)

#             if face_id not in face_size_queues:
#                 face_size_queues[face_id] = np.zeros((face_center_window, 2))
#             face_size_queue = face_size_queues[face_id]
#             face_size_queue[:-1] = face_size_queue[1:]
#             face_size_queue[-1] = face_width, face_height
#             history = min(face_center_window, i + 1)
#             mean_face_width, mean_face_height = face_size_queue[-history:].mean(axis=0)  # moving average of face size
#             margin = 80
#             frame_cropped = extract_face_with_margin(
#                 frame, (center_x, center_y), mean_face_width, mean_face_height, margin
#             )
#             if np.prod(frame_cropped.shape) == 0:
#                 continue

#             frame_resized = cv2.resize(
#                 frame_cropped,
#                 (output_image_size, output_image_size),
#                 interpolation=cv2.INTER_CUBIC,
#             )
#             face_frames[face_id].append(frame_resized)

#     return face_frames


# def detect_face_blazeface_full(frames, prob_thresh=0.8, output_image_size=224, face_center_window=30):
#     mp_face_detection = mp.solutions.face_detection
#     face_frames = defaultdict(list)
#     face_size_queues = {}
#     face_ids = {}

#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=prob_thresh) as face_detection:
#         for i, frame in enumerate(frames):
#             frame.flags.writeable = False
#             results = face_detection.process(frame)
#             if not results.detections:  # no face detected
#                 continue

#             height, width, _ = frame.shape
#             for detection in results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 nose = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
#                 nose_x, nose_y = int(nose.x * width), int(nose.y * height)
#                 center_x, center_y = nose_x, nose_y
#                 bbox_width, bbox_height = int(bbox.width * width), int(bbox.height * height)

#                 # Face tracking from center of face
#                 for face_id, face_center in face_ids.items():
#                     if (face_center[0] - center_x) ** 2 + (
#                         face_center[1] - center_y
#                     ) ** 2 < bbox_width**2 + bbox_height**2:
#                         break
#                 else:
#                     face_id = len(face_ids)
#                 face_ids[face_id] = (center_x, center_y)

#                 if face_id not in face_size_queues:
#                     face_size_queues[face_id] = np.zeros((face_center_window, 2))
#                 face_size_queue = face_size_queues[face_id]
#                 face_size_queue[:-1] = face_size_queue[1:]
#                 face_size_queue[-1] = bbox_width, bbox_height
#                 history = min(face_center_window, i + 1)
#                 mean_face_width, mean_face_height = face_size_queue[-history:].mean(
#                     axis=0
#                 )  # moving average of face size
#                 # frame[center_y-2:center_y+2, center_x-2:center_x+2, :] = (255, 0, 0)

#                 frame_cropped = extract_face_with_margin(
#                     frame,
#                     (center_x, center_y),
#                     mean_face_width,
#                     mean_face_height,
#                     margin=80,
#                 )
#                 if np.prod(frame_cropped.shape) == 0:
#                     continue

#                 if len(face_frames[face_id]):

#                     frame_cropped = cv2.resize(
#                         frame_cropped,
#                         face_frames[face_id][-1].shape[1::-1],
#                         interpolation=cv2.INTER_CUBIC,
#                     )

#                 face_frames[face_id].append(frame_cropped)

#     return face_frames


# def detect_face_retinaface_full(frames, prob_thresh=0.7, output_image_size=224, face_center_window=30):
#     global _RETINAFACE
#     if _RETINAFACE is None:
#         from retinaface.pre_trained_models import get_model

#         _RETINAFACE = get_model("resnet50_2020-07-20", max_size=2048)
#         _RETINAFACE.eval()
#         if torch.cuda.is_available():
#             _RETINAFACE.device = torch.device("cuda")
#             _RETINAFACE.model = _RETINAFACE.model.to(_RETINAFACE.device)

#     face_frames = defaultdict(list)
#     face_size_queues = {}
#     face_ids = {}

#     with torch.no_grad():
#         for i, frame in enumerate(tqdm(frames, desc="Detecting faces")):
#             frame.flags.writeable = False
#             annotations = _RETINAFACE.predict_jsons(frame, confidence_threshold=prob_thresh)
#             if len(annotations) == 0:
#                 continue

#             for detection in annotations:
#                 bbox = detection["bbox"]
#                 score = detection["score"]
#                 center_x, center_y = np.mean(detection["landmarks"], axis=0)
#                 bbox_width = int(bbox[2] - bbox[0])
#                 bbox_height = int(bbox[3] - bbox[1])

#                 # Face tracking from center of face
#                 for face_id, face_center in face_ids.items():
#                     if (face_center[0] - center_x) ** 2 + (
#                         face_center[1] - center_y
#                     ) ** 2 < bbox_width**2 + bbox_height**2:
#                         break
#                 else:
#                     face_id = len(face_ids)
#                 face_ids[face_id] = (center_x, center_y)

#                 if face_id not in face_size_queues:
#                     face_size_queues[face_id] = np.zeros((face_center_window, 2))
#                 face_size_queue = face_size_queues[face_id]
#                 face_size_queue[:-1] = face_size_queue[1:]
#                 face_size_queue[-1] = bbox_width, bbox_height
#                 history = min(face_center_window, i + 1)
#                 mean_face_width, mean_face_height = face_size_queue[-history:].mean(
#                     axis=0
#                 )  # moving average of face size
#                 # frame[center_y-2:center_y+2, center_x-2:center_x+2, :] = (255, 0, 0)

#                 frame_cropped = extract_face_with_margin(
#                     frame,
#                     (center_x, center_y),
#                     mean_face_width,
#                     mean_face_height,
#                     margin=80,
#                 )
#                 if np.prod(frame_cropped.shape) == 0:
#                     continue

#                 frame_resized = cv2.resize(
#                     frame_cropped,
#                     (output_image_size, output_image_size),
#                     interpolation=cv2.INTER_CUBIC,
#                 )

#                 face_frames[face_id].append(frame_resized)

#     return face_frames
