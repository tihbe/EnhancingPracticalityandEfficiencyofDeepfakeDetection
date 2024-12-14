import os
import numpy as np
from tqdm import tqdm
import cv2
from shutil import copyfile
from fire import Fire
from dfdetect.data_loaders import DFDC
import dfdetect.preprocessing.face_detection as fd
from dfdetect.utils import frames_to_video, crop_frames, Slurm
from typing import Optional
from itertools import islice
from dfdetect.config import Paths

# TODO
# Keep audio from original video ?


def main(
    old_path=os.environ.get("DFDC_DATASET_PATH"),
    new_path=os.environ.get("DFDC_PREPROCESSED_DATASET_PATH", "dfdc_preprocessed"),
    face_detection_method="blazeface",  # one of: "blazeface", "mtcnn", retinaface
    store_as_video=True,
    compressed=True,
    target_fps: Optional[int] = None,
    slurm_multiprocessing=Slurm.is_active(),
    is_test=False,
):
    """Preprocessing script to extract faces from videos"""
    np.random.seed(0x1B)

    dataset = DFDC(old_path, is_test=is_test)
    os.makedirs(new_path, exist_ok=True)

    if slurm_multiprocessing is False or Slurm.is_first_task():
        metadata_fname = "labels.csv" if is_test else "metadata.json"
        copyfile(
            os.path.join(old_path, metadata_fname),
            os.path.join(new_path, metadata_fname),
        )

    if slurm_multiprocessing:
        block_size = int(np.ceil(len(dataset) / Slurm.task_count()))
        task_id = Slurm.task_id()
        target_videos = np.arange(
            task_id * block_size, min((task_id + 1) * block_size, len(dataset))
        )
    else:
        target_videos = np.arange(0, len(dataset))

    for i in tqdm(target_videos, desc="Converting videos to faces"):
        frames, label = dataset[i]
        h, w, _ = frames[0].shape

        if face_detection_method == "blazeface":
            bboxes = fd.detect_face_blazeface(frames)
        elif face_detection_method == "mtcnn":
            bboxes = fd.detect_face_mtcnn(frames)
        elif face_detection_method == "retinaface":
            bboxes = fd.detect_face_retinaface(frames)
        else:
            raise f"{face_detection_method} is not a supported face detection method"

        tracked_faces = fd.face_tracking(bboxes)
        face_frames = {}

        for face_id, face_bboxes in tracked_faces.items():
            if (
                sum(bbox != None for bbox in face_bboxes) < 10
            ):  # Require at least 10 frames of face detection
                continue
            face_width, face_height = fd.max_size_bbox(face_bboxes)
            face_bboxes = fd.smooth_bboxes_centers(face_bboxes)
            face_width = int(face_width * 1.15)  # Add 15% padding
            face_height = int(face_height * 1.15)
            face_bboxes = fd.recenter_bboxes(face_bboxes, face_width, face_height, w, h)
            face_frames[face_id] = crop_frames(frames, face_bboxes)
            # face_frames[face_id] = crop_and_stabilize2(frames, face_bboxes)

        video_original_name = dataset.get_filename(i)
        for face_id, face_frames in face_frames.items():
            fps = dataset.last_fps
            if target_fps is not None and fps > target_fps:
                face_frames = islice(face_frames, 0, None, int(fps / target_fps))
                fps = target_fps

            if store_as_video:
                path = os.path.join(
                    new_path, video_original_name.replace(".", f"_{face_id}.")
                )
                _, file_extension = os.path.splitext(path)
                frames_to_video(
                    face_frames,
                    path if compressed else path.replace(file_extension, ".avi"),
                    fps,
                    codec="vp09" if compressed else "RGBA",
                )
            else:  # Store each frames individually
                video_path = os.path.join(new_path, video_original_name)
                os.makedirs(video_path, exist_ok=True)
                for j, face_frame in enumerate(face_frames):
                    face_frame_path = os.path.join(video_path, f"{face_id}_{j}.png")
                    cv2.imwrite(
                        face_frame_path, cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                    )


if __name__ == "__main__":
    Fire(main)
