import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.auto import tqdm
from itertools import chain
from facenet_pytorch import MTCNN
from skimage.color import rgb2gray
from vidstab import VidStab
from collections import defaultdict

DATASET_PATH = os.environ.get("DFDC_DATASET_PATH")
SAMPLES_PATH = os.path.join(DATASET_PATH, "train_sample_videos")


def main():
    plt.rcParams["figure.figsize"] = [10, 5]

    desc = pd.read_json(os.path.join(SAMPLES_PATH, "metadata.json")).transpose()

    print(desc.head())

    if classif := False:

        # Part 1: Since the dataset is incomplete
        # Keep only pairs of videos that exists in the metadata.json file
        mask = np.empty(len(desc), dtype=bool)
        existing_files = os.listdir(SAMPLES_PATH)
        for i, (f_name, row) in enumerate(desc.iterrows()):
            m = f_name in existing_files
            if row.label == "FAKE":
                m &= row.original in existing_files
            mask[i] = m
        desc = desc[mask]

        # Part 2: Iterate through the fake videos and their original
        # And compute score using the new spectral method
        fake_videos = desc[desc.label == "FAKE"]
        mtcnn = MTCNN(160, margin=30, keep_all=True, post_process=False)
        for f_name, row in tqdm(fake_videos.iterrows()):
            fake_sample = f_name
            real_sample = row.original

            try:
                fake_frames, f_fps = read_sample(fake_sample)
                real_frames, r_fps = read_sample(real_sample)

                fake_score = compute_mean_spectral_movement(fake_frames, mtcnn)
                real_score = compute_mean_spectral_movement(real_frames, mtcnn)
            except:
                print(f"Failed to process {fake_sample}")
                continue

            print(np.abs(fake_score), np.abs(real_score))
            with open("mean_spectral_movement_score_norm.csv", "a") as f_hndl:
                f_hndl.write(
                    f"{fake_sample},{real_sample},{np.abs(fake_score)},{np.abs(real_score)}\n"
                )

        exit(0)

    fake_sample = "etmcruaihe.mp4"
    real_sample = "afoovlsmtx.mp4"

    fake_frames, f_fps = read_sample(fake_sample)
    real_frames, r_fps = read_sample(real_sample)

    # Plot first frames of fake and real images
    if False:
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        axs[0].imshow(real_frames[0])
        axs[0].set_title("Real image")
        axs[1].imshow(fake_frames[0])
        axs[1].set_title("Fake image")
        axs[0].axis("off")
        axs[1].axis("off")

    # Face detection using MTCNN
    mtcnn = MTCNN(160, margin=30, keep_all=True, post_process=False)

    face_to_spectral_video_duo(
        real_frames, fake_frames, mtcnn, "spectral_real_fake.mp4"
    )

    # face_to_video(fake_frames, mtcnn, "fake_face.mp4", f_fps)
    # face_to_video(real_frames, mtcnn, "real_face.mp4", r_fps)

    # face_to_spectral_video(fake_frames, mtcnn, "fake_face_spectral.mp4", 5)
    # face_to_spectral_video(real_frames, mtcnn, "real_face_spectral.mp4", 5)

    # f_score = compute_mean_spectral_movement(fake_frames, mtcnn)
    # print(f_score)
    # r_score = compute_mean_spectral_movement(real_frames, mtcnn)
    # print(r_score)

    # print((f_score), (r_score))
    # print(np.abs(f_score), np.abs(r_score))

    exit(0)


def dead_spectral_code():
    # fig, ax = plt.subplots()
    # ax.scatter(spectral_frame_flat.real, spectral_frame_flat.imag)
    # fig.show()

    spectral_amp = np.abs(spectral_features)
    spectral_pha = np.angle(spectral_features)
    log_spectral_amp_norm = np.log(spectral_amp + 1e-12)
    gmin, gmax = log_spectral_amp_norm.min(), log_spectral_amp_norm.max()
    log_spectral_amp_norm = (log_spectral_amp_norm - gmin) / (gmax - gmin)
    fig, ax = plt.subplots(1, 2)
    # ax.hist(log_spectral_amp_norm.flatten())
    ax[0].imshow(log_spectral_amp_norm)
    ax[1].imshow(spectral_pha)
    ax[1].axis("off")

    plt.show()
    exit(0)

    if False:
        gmin, gmax = np.inf, -np.inf
        for f in chain(fake_frames, real_frames):
            spectral_amp = np.abs(np.fft.fft2(f, axes=(0, 1)))
            log_spectral_amp = np.log(spectral_amp + 1e-12)
            if log_spectral_amp.min() < gmin:
                gmin = log_spectral_amp.min()
            if log_spectral_amp.max() > gmax:
                gmax = log_spectral_amp.max()
        print(gmin, gmax)
    else:  # Result
        gmin, gmax = -26.380846, 19.690151

    log_spectral_amp_norm_last = None
    for i, f in enumerate(tqdm(fake_frames)):
        spectral_amp = np.abs(np.fft.fft2(f, axes=(0, 1)))
        log_spectral_amp_norm = np.log(spectral_amp + 1e-12)
        log_spectral_amp_norm = (
            log_spectral_amp_norm - log_spectral_amp_norm.min()
        ) / (log_spectral_amp_norm.max() - log_spectral_amp_norm.min())

        if log_spectral_amp_norm_last is not None:
            diff = np.abs(log_spectral_amp_norm_last - log_spectral_amp_norm)
            mask = diff > 0.1
            thresholded = np.where(diff > 0.1, np.ones_like(diff), np.zeros_like(diff))
            fig, ax = plt.subplots()
            ax.imshow(thresholded)
            fig.savefig(os.path.join("fake_image_hists", f"{i}.png"), facecolor="white")
            plt.close(fig)
        log_spectral_amp_norm_last = log_spectral_amp_norm

    for i, f in enumerate(tqdm(real_frames)):
        spectral_amp = np.abs(np.fft.fft2(f, axes=(0, 1)))
        log_spectral_amp_norm = np.log(spectral_amp + 1e-12)
        log_spectral_amp_norm = (
            log_spectral_amp_norm - log_spectral_amp_norm.min()
        ) / (log_spectral_amp_norm.max() - log_spectral_amp_norm.min())

        if log_spectral_amp_norm_last is not None:
            diff = np.abs(log_spectral_amp_norm_last - log_spectral_amp_norm)
            mask = diff > 0.1
            thresholded = np.where(diff > 0.1, np.ones_like(diff), np.zeros_like(diff))
            fig, ax = plt.subplots()
            ax.imshow(thresholded)
            fig.savefig(os.path.join("real_image_hists", f"{i}.png"), facecolor="white")
            plt.close(fig)
        log_spectral_amp_norm_last = log_spectral_amp_norm


def read_sample(sample_id, /, nb_frames=np.inf):
    """Read video by id and return array of frames up to nb_frames"""
    cap = cv2.VideoCapture(os.path.join(SAMPLES_PATH, sample_id))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cap.get(cv2.CAP_PROP_FOURCC)
    frames = []
    while cap.isOpened() and nb_frames > 0:
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        nb_frames -= 1
    cap.release()
    return frames, fps, codec


def mpl_to_npy(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def spectral_to_video(frames, mtcnn, video_name):
    o_video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), 5, (1000, 500)
    )

    for frame in tqdm(frames):
        faces = mtcnn(frame)
        batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(frame, landmarks=True)

        # Extract faces
        faces = mtcnn.extract(frame, batch_boxes, None)
        faces = faces.permute(0, 2, 3, 1)
        # face = faces[0].int().numpy()

        # Extract fourier of face 0
        # gray_face = rgb2gray(faces[0].numpy() / 255)
        face = faces[0].numpy() / 255
        spectral_features = np.fft.fftshift(np.fft.fft2(face, axes=(0, 1)), axes=(0, 1))
        spectral_features_flat = spectral_features.flatten()
        fig, ax = plt.subplots()
        ax.scatter(spectral_features_flat.real, spectral_features_flat.imag)
        xlim = [-2500, 2500]
        ylim = [-1300, 1300]
        xlim[0] = np.minimum(xlim[0], spectral_features_flat.real.min())
        ylim[0] = np.minimum(ylim[0], spectral_features_flat.imag.min())
        xlim[1] = np.maximum(xlim[1], spectral_features_flat.real.max())
        ylim[1] = np.maximum(ylim[1], spectral_features_flat.imag.max())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        o_frame = mpl_to_npy(fig)
        o_video.write(o_frame)

    cv2.destroyAllWindows()
    o_video.release()


def preprocess_faces(
    frames, mtcnn, stabilizing_window=30, ouput_image_size=160, prob_thresh=0.9
):
    """Stabilize, rescale and center the faces of frames based on the centermost landmark"""
    # face_size_queue = np.zeros((30, 2))
    # stabilizer = VidStab()
    stabilizers = {}
    face_size_queues = {}  # np.zeros((30, 2))
    stabilized_face_frames = defaultdict(list)

    for i, frame in enumerate(tqdm(frames, desc="Preprocessing faces")):
        batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(frame, landmarks=True)

        if (
            batch_boxes is None or (nb_faces := len(batch_boxes)) == 0
        ):  # no face detected
            continue

        for face_idx in range(nb_faces):
            if batch_probs[face_idx] < prob_thresh:
                continue
            if face_idx not in stabilizers:
                stabilizers[face_idx] = VidStab()
                face_size_queues[face_idx] = np.zeros((stabilizing_window, 2))
            face_size_queue = face_size_queues[face_idx]
            face_landmarks = batch_landmarks[face_idx]
            dist_to_center = (
                face_landmarks - face_landmarks.mean(axis=0, keepdims=True)
            ) ** 2
            center_landmark = np.argmin(dist_to_center.sum(axis=1))
            img_center = face_landmarks[center_landmark]
            bbox = batch_boxes[face_idx]
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size_queue[:-1] = face_size_queue[1:]
            face_size_queue[-1] = face_width, face_height
            mean_face_width, mean_face_height = face_size_queue[-i:].mean(axis=0)

            margin = 80
            bbox = [
                int(max(img_center[0] - mean_face_width / 2 - margin / 2, 0)),
                int(max(img_center[1] - mean_face_height / 2 - margin / 2, 0)),
                int(
                    min(
                        img_center[0] + mean_face_width / 2 + margin / 2, frame.shape[0]
                    )
                ),
                int(
                    min(
                        img_center[1] + mean_face_height / 2 + margin / 2,
                        frame.shape[1],
                    )
                ),
            ]
            frame_cropped = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if np.prod(frame_cropped.shape) == 0:
                continue

            frame_resized = cv2.resize(
                cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR),
                (ouput_image_size, ouput_image_size),
                interpolation=cv2.INTER_CUBIC,
            )

            frame_stabilized = stabilizers[face_idx].stabilize_frame(
                input_frame=frame_resized, smoothing_window=stabilizing_window
            )
            stabilized_face_frames[face_idx].append(frame_stabilized)

    for face_idx in list(stabilizers.keys()):
        if len(stabilized_face_frames[face_idx]) <= stabilizing_window:
            del stabilized_face_frames[face_idx]
            del stabilizers[face_idx]

    for i in range(stabilizing_window):
        for face_idx in stabilizers.keys():
            # empty frames buffer
            frame = stabilizers[face_idx].stabilize_frame(
                input_frame=None, smoothing_window=stabilizing_window
            )

            if frame is not None:
                stabilized_face_frames[face_idx].append(frame)

    for face_idx in stabilized_face_frames.keys():
        stabilized_face_frames[face_idx] = stabilized_face_frames[face_idx][
            stabilizing_window:
        ]

    return stabilized_face_frames


def face_to_video(frames, mtcnn, video_name, fps=5):
    """Call preprocess_faces to extract faces and export to a mp4 video"""
    if ".mp4" not in video_name:
        video_name = video_name + ".mp4"
    o_video = None

    stabilized_face_frames = preprocess_faces(frames, mtcnn)

    for frame in stabilized_face_frames:
        # Output to video
        if o_video is None:
            o_video = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[:2]
            )

        o_video.write(frame)

    cv2.destroyAllWindows()
    o_video.release()


def face_to_spectral_video(frames, mtcnn, video_name, fps=5):
    """Call preprocess_faces to extract faces, extract spectral features with FFT, plot with mpl and export to a mp4 video"""

    if ".mp4" not in video_name:
        video_name = video_name + ".mp4"
    o_video = None

    stabilized_face_frames = preprocess_faces(frames, mtcnn)

    for frame in tqdm(stabilized_face_frames):
        spectral_features = np.fft.fftshift(
            np.fft.fft2(frame / 255, axes=(0, 1)), axes=(0, 1)
        )
        spectral_features_flat = spectral_features.flatten()
        fig, ax = plt.subplots()
        ax.scatter(spectral_features_flat.real, spectral_features_flat.imag)
        xlim = [-2500, 2500]
        ylim = [-1300, 1300]
        xlim[0] = np.minimum(xlim[0], spectral_features_flat.real.min())
        ylim[0] = np.minimum(ylim[0], spectral_features_flat.imag.min())
        # xlim[1] = np.maximum(xlim[1], spectral_features_flat.real.max())
        ylim[1] = np.maximum(ylim[1], spectral_features_flat.imag.max())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        o_frame = mpl_to_npy(fig)

        cv2.imshow(video_name, o_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Output to video
        if o_video is None:
            o_video = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                o_frame.shape[:2][::-1],
            )

        o_video.write(o_frame)

    cv2.destroyAllWindows()
    o_video.release()


def face_to_spectral_video_duo(frames_real, frames_fake, mtcnn, video_name, fps=5):
    """Call preprocess_faces to extract faces, extract spectral features with FFT, plot with mpl and export to a mp4 video"""

    if ".mp4" not in video_name:
        video_name = video_name + ".mp4"
    o_video = None

    stabilized_face_frames_real = preprocess_faces(frames_real, mtcnn)
    stabilized_face_frames_fake = preprocess_faces(frames_fake, mtcnn)

    for frame_real, frame_fake in zip(
        tqdm(stabilized_face_frames_real), stabilized_face_frames_fake
    ):
        spectral_features_real = np.fft.fftshift(
            np.fft.fft2(frame_real / 255, axes=(0, 1)), axes=(0, 1)
        )
        spectral_features_fake = np.fft.fftshift(
            np.fft.fft2(frame_fake / 255, axes=(0, 1)), axes=(0, 1)
        )
        spectral_features_flat_real = spectral_features_real.flatten()
        spectral_features_flat_fake = spectral_features_fake.flatten()
        fig, ax = plt.subplots()
        ax.scatter(
            spectral_features_flat_real.real,
            spectral_features_flat_real.imag,
            label="Real",
        )
        ax.scatter(
            spectral_features_flat_fake.real,
            spectral_features_flat_fake.imag,
            label="Fake",
        )
        xlim = [-2500, 2500]
        ylim = [-1300, 1300]
        xlim[0] = np.minimum(xlim[0], spectral_features_flat_real.real.min())
        ylim[0] = np.minimum(ylim[0], spectral_features_flat_real.imag.min())
        # xlim[1] = np.maximum(xlim[1], spectral_features_flat.real.max())
        ylim[1] = np.maximum(ylim[1], spectral_features_flat_real.imag.max())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc="lower right")
        o_frame = mpl_to_npy(fig)

        cv2.imshow(video_name, o_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Output to video
        if o_video is None:
            o_video = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                o_frame.shape[:2][::-1],
            )

        o_video.write(o_frame)

    cv2.destroyAllWindows()
    o_video.release()


def compute_mean_spectral_movement(frames, mtcnn):
    """Compute mean vector movement from the spectral features of each frequency over every pair of frames at time t and t+1"""

    stabilized_face_frames = preprocess_faces(frames, mtcnn)
    vector_distances = []
    for frame_0, frame_1 in zip(
        tqdm(stabilized_face_frames[:-1]), stabilized_face_frames[1:]
    ):
        spectral_features_0 = np.fft.fftshift(
            np.fft.fft2(frame_0 / 255, axes=(0, 1)), axes=(0, 1)
        )
        spectral_features_1 = np.fft.fftshift(
            np.fft.fft2(frame_1 / 255, axes=(0, 1)), axes=(0, 1)
        )

        vector_distances.append(
            np.linalg.norm(spectral_features_0 - spectral_features_1)
        )

    return np.mean(vector_distances)


if __name__ == "__main__":
    main()
