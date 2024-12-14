import io
import os

import cv2
import dfdetect.preprocessing.face_detection as fd
import dfdetect.utils as utils
import numpy as np
from dfdetect.data_loaders import DFDC
from dfdetect.utils import Slurm
from fire import Fire
from tqdm import tqdm


def main(
    old_path=os.environ.get("DFDC_DATASET_PATH"),
    new_path=os.environ.get("DFDC_PREPROCESSED_DATASET_PATH", "dfdc_preprocessed"),
    slurm_multiprocessing=Slurm.is_active(),
    is_test=False,
    target_ratio=0.5,  # minimum ratio of change in the pixel difference
):
    """Preprocessing script that only outputs a single frame from each DFDC video"""
    np.random.seed(0x1B)

    dataset = DFDC(old_path, is_test=is_test)
    os.makedirs(new_path, exist_ok=True)

    metadata_fname = os.path.join(new_path, "labels.csv")
    if slurm_multiprocessing is False or Slurm.is_first_task():
        with io.open(metadata_fname, "w") as f:
            f.write("dfdc_id,file_name,label\n")

    if slurm_multiprocessing:
        block_size = int(np.ceil(len(dataset) / Slurm.task_count()))
        task_id = Slurm.task_id()
        target_videos = np.arange(
            task_id * block_size, min((task_id + 1) * block_size, len(dataset))
        )
    else:
        target_videos = np.arange(0, len(dataset))

    for index in tqdm(target_videos, desc="Converting videos to faces"):
        meta = dataset.desc.iloc[index]
        output_frames = []
        error_loading = True
        if meta["label"] == "FAKE" and not is_test:
            """In training, we use the original video to filter our fake faces from real faces in fake videos"""
            original_name = meta["original"]
            original_path = os.path.join(os.path.dirname(meta["path"]), original_name)
            real_frames = utils.video_to_frames(original_path)[0]
            fake_frames = utils.video_to_frames(meta["path"])[0]
            for real_frame, fake_frame in zip(real_frames, fake_frames):
                error_loading = False
                bboxes = fd.waterfall(real_frame)
                for bbox in bboxes:
                    bbox.recast()
                    if bbox.width() * bbox.height() == 0:
                        continue
                    real_face, fake_face = list(
                        utils.crop_frames([real_frame, fake_frame], [bbox, bbox])
                    )
                    change_ratio = np.count_nonzero(real_face - fake_face) / np.product(
                        real_face.shape
                    )
                    if change_ratio > target_ratio:
                        output_frames.append(fake_face)
                if len(output_frames) > 0:
                    break
        else:  # REAL or testing
            real_frames = utils.video_to_frames(meta["path"])[0]
            for real_frame in real_frames:
                error_loading = False
                bboxes = fd.waterfall(real_frame)
                for n in range(len(bboxes)):
                    bbox = bboxes[n]
                    bbox.recast()
                    if bbox.width() * bbox.height() == 0:
                        del bboxes[n]
                real_faces = list(utils.crop_frames([real_frame] * len(bboxes), bboxes))
                if len(real_faces) > 0:
                    if is_test:  # Keep only 1 image per video in test
                        output_frames.append(real_faces[0])
                    else:
                        output_frames += real_faces
                    break

        if is_test and len(output_frames) == 0:
            # Preprocess test frame by cropping a 100x100 block in the middle for the frames where face is undetectable
            meta = dataset.desc.iloc[index]
            video_original_name = dataset.get_filename(index)
            _, file_extension = os.path.splitext(video_original_name)
            image_path = os.path.join(
                new_path, video_original_name.replace(file_extension, "_0.png")
            )
            frames = list(utils.video_to_frames(meta["path"])[0])
            if len(frames) == 0:
                print("Error while loading video:", meta["path"], meta["label"])
                return
            mid_frame = len(frames) // 2
            frame = frames[mid_frame]
            h, w, c = frame.shape
            sh, sw = h // 2 - 50, w // 2 - 50
            face = frame[sh : sh + 100, sw : sw + 100, :]
            cv2.imwrite(image_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            with io.open(metadata_fname, "a") as f:
                f.write(f"{video_original_name},{image_path},{meta['label']}\n")

        if error_loading:
            print("Error while loading video:", meta["path"], meta["label"])
            continue

        if len(output_frames) == 0:
            print("Error while processing video:", meta["path"])
        else:
            video_original_name = dataset.get_filename(index)
            _, file_extension = os.path.splitext(video_original_name)
            image_path = os.path.join(
                new_path, video_original_name.replace(file_extension, ".png")
            )

            for i in range(len(output_frames)):
                try:
                    c_path = image_path.replace(".png", f"_{i}.png")
                    cv2.imwrite(
                        c_path, cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2BGR)
                    )

                    with io.open(metadata_fname, "a") as f:
                        f.write(f"{video_original_name},{c_path},{meta['label']}\n")
                except Exception as e:
                    print(e)
                    print("Error while saving for video:", meta["path"])
                    continue


if __name__ == "__main__":
    Fire(main)
