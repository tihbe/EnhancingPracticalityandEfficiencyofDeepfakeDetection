import os
from shutil import copyfile

import cv2
import dfdetect.preprocessing.face_detection as fd
import dfdetect.utils as utils
import numpy as np
from dfdetect.config import Paths
from dfdetect.data_loaders import CelebDFV2
from dfdetect.utils import Slurm
from fire import Fire
from tqdm import tqdm


def main(
    slurm_multiprocessing=Slurm.is_active(),
    test=False,
    target_ratio=0.5,  # minimum ratio of change in the pixel difference
):
    """Preprocessing script that only outputs a single frame from each DFDC video"""
    np.random.seed(0x1B)

    dataset = CelebDFV2(Paths.CelebDFV2.dataset_path, is_train=not test)
    new_directory = Paths.CelebDFV2.preprocessed_path

    if slurm_multiprocessing is False or Slurm.is_first_task():
        os.makedirs(new_directory, exist_ok=True)
        os.makedirs(os.path.join(new_directory, "YouTube-real"), exist_ok=True)
        os.makedirs(os.path.join(new_directory, "Celeb-synthesis"), exist_ok=True)
        os.makedirs(os.path.join(new_directory, "Celeb-real"), exist_ok=True)
        with open(
            os.path.join(Paths.CelebDFV2.dataset_path, "List_of_testing_videos.txt"),
            "r",
        ) as f:
            lines = f.readlines()
            lines = [line.replace(".mp4", "_0.png") for line in lines]
        with open(os.path.join(new_directory, "List_of_testing_videos.txt"), "w") as f:
            f.writelines(lines)
    else:
        import time

        time.sleep(500)  # Making sure the directories exist

    if slurm_multiprocessing:
        block_size = int(np.ceil(len(dataset) / Slurm.task_count()))
        task_id = Slurm.task_id()
        target_videos = np.arange(
            task_id * block_size, min((task_id + 1) * block_size, len(dataset))
        )
    else:
        target_videos = np.arange(0, len(dataset))

    for index in tqdm(target_videos, desc="Converting videos to images"):
        video_path = dataset.files[index]
        label = dataset.labels[index]
        output_frames = []
        error_loading = True

        filename = os.path.basename(video_path)
        directory = os.path.dirname(video_path)
        parts = filename.split("_")

        if not test and label == 0:  # FAKE training sample
            """In training, we use the original video to filter our fake faces from real faces in fake videos"""

            original_file = os.path.join(
                directory.replace("Celeb-synthesis", "Celeb-real"),
                "_".join((parts[0], parts[-1])),
            )
            assert os.path.exists(original_file)
            real_frames = utils.video_to_frames(original_file)[0]
            fake_frames = utils.video_to_frames(video_path)[0]
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
        else:  # real video or testing
            real_frames = utils.video_to_frames(video_path)[0]
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
                    if test:  # Keep only 1 image per video in test
                        output_frames.append(real_faces[0])
                    else:
                        output_frames += real_faces
                    break

        if test and len(output_frames) == 0:
            # Preprocess test frame by cropping a 100x100 block in the middle for the frames where face is undetectable
            frames = list(utils.video_to_frames(video_path)[0])
            mid_frame = len(frames) // 2
            frame = frames[mid_frame]
            h, w, c = frame.shape
            sh, sw = h // 2 - 50, w // 2 - 50
            face = frame[sh : sh + 100, sw : sw + 100, :]
            output_frames.append(face)

        if error_loading:
            print("Error while loading video:", video_path)
            continue

        if len(output_frames) == 0:
            print("Error while processing video:", video_path)
        else:
            image_path = video_path.replace(
                Paths.CelebDFV2.dataset_path, new_directory
            )  # TODO Not great but should work for now :)
            image_path = image_path.replace(".mp4", ".png")
            for i in range(len(output_frames)):
                try:
                    c_path = image_path.replace(".png", f"_{i}.png")
                    cv2.imwrite(
                        c_path, cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2BGR)
                    )
                except Exception as e:
                    print(e)
                    print("Error while saving for video:", c_path)
                    continue


if __name__ == "__main__":
    Fire(main)
