import os
import cv2
import numpy as np
import torch
from typing import List, Optional
from dfdetect.preprocessing.face_detection import BBox
import torchvision.transforms.functional as F


def frames_to_video(frames, path, fps, codec="vp09", quality=100):
    """Export a list (or generator) of frames into a video"""
    o_video = None
    fourcc = cv2.VideoWriter_fourcc(*codec)
    try:
        for frame in frames:
            if o_video is None:
                h, w, _ = frame.shape
                o_video = cv2.VideoWriter(path, fourcc, fps, (w, h))
                o_video.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
            o_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if o_video is not None:
            cv2.destroyAllWindows()
            o_video.release()


def hash_dict(d):
    """Hash a dictionary"""
    return hash(str(sorted(d.items())))


def video_to_frames(video_path, /, nb_frames=np.inf):
    """Read video by path and return generator of frames up to nb_frames, with fps and codec"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    def generator(nb_frames):
        while cap.isOpened() and nb_frames > 0:
            ret, frame = cap.read()
            if frame is None or not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
            nb_frames -= 1
        cap.release()

    return generator(nb_frames), fps, codec


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return fps


def crop_frames(frames: List[np.ndarray], bboxes: List[Optional[BBox]]):
    return (
        f[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
        for f, bbox in zip(frames, bboxes)
        if bbox is not None
    )


class Slurm:
    @staticmethod
    def job_id():
        return int(os.environ["SLURM_JOB_ID"])

    @staticmethod
    def is_active() -> bool:
        return os.environ.get("SLURM_JOB_ID", None) is not None

    @staticmethod
    def task_id() -> int:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])

    @staticmethod
    def min_task() -> int:
        return int(os.environ["SLURM_ARRAY_TASK_MIN"])

    @staticmethod
    def task_count() -> int:
        return int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    @staticmethod
    def is_first_task() -> bool:
        return Slurm.task_id() == Slurm.min_task()

    @staticmethod
    def cpu_count(default: int = os.cpu_count()) -> int:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", default))

    @staticmethod
    def gpu_count(default: int = torch.cuda.device_count()) -> int:
        selected_gpus = os.environ.get("SLURM_JOB_GPUS", None)
        if selected_gpus is None:
            return default
        else:
            return len(selected_gpus.split(","))


class FrameBasedTransforms:
    """Takes a transform and apply it to every frames of a video"""

    def __init__(self, frame_transform):
        self.frame_transform = frame_transform

    def __call__(self, frames):
        frames = [self.frame_transform(img) for img in frames]
        return torch.stack(frames, dim=0) if len(frames) > 0 else torch.empty(0)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class CropResize:
    """Center crop and resize to keep aspect ratio"""

    def __init__(self, target_size):
        self.size = target_size

    def center_crop(self, frame, size):
        h, w, _ = frame.shape
        startx = w // 2 - size // 2
        starty = h // 2 - size // 2
        return frame[starty : starty + size, startx : startx + size]

    def __call__(self, frame):
        h, w, _ = frame.shape
        min_len = min(h, w)
        frame = self.center_crop(frame, min_len)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return frame


def rgb_to_ycc(image: np.ndarray) -> np.ndarray:
    """Convert RGB to YCbCr"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale"""
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_scale[:, :, np.newaxis]


def halve_frames(frames):
    return frames[::2]


def collate_zero_pad(batch):
    """Collate a batch of videos with zero padding, taking the sequence length of the batch to be the maximum sample duration"""

    X, y = list(zip(*batch))
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    return X, torch.tensor(y)


def dct(x: torch.tensor, norm: Optional[str] = None) -> torch.tensor:
    """DCT Transform - adapted from https://github.sre.pub/zh217/torch-dct/blob/master/torch_dct/_dct.py"""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(frame: torch.tensor, norm: Optional[str] = None) -> torch.tensor:
    """2D DCT Transform - adapted from https://github.sre.pub/zh217/torch-dct/blob/master/torch_dct/_dct.py"""
    X1 = dct(frame, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def get_pl_loggers(exp_id, configs: dict):
    """Set up a neptune and tensorboard loggers with configuration dictionary.
    The configuration dictionary must at least contain a project_name key. If a debug flag is set, the loggers will be empty
    """
    if "debug" in configs.keys() and configs["debug"] is True:
        return False

    from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

    project_name = configs["project_name"]
    tags = []
    loggers = []
    for param, value in configs.items():
        if type(value) is bool:
            if value:
                tags.append(str(param))
            else:
                tags.append("no_" + str(param))
        elif type(value) is str:
            tags.append(value)

    comet_logger = CometLogger(
        project_name=project_name,
        api_key=os.environ["COMET_API_KEY"],
        rest_api_key=os.environ["COMET_API_KEY"],
        experiment_name=exp_id,
    )
    comet_logger.experiment.add_tags(tags)
    comet_logger.experiment.log_parameters(configs)
    loggers.append(comet_logger)
    os.makedirs("tb_logs", exist_ok=True)
    tensorboard_logger = TensorBoardLogger(save_dir="tb_logs", name=project_name)
    loggers.append(tensorboard_logger)
    return loggers


def compute_running_stats(dataset):
    """Compute mean and variance of the dataset when transformed from RGB to YCC"""
    from tqdm.auto import tqdm

    Xs = []
    for X, _ in tqdm(dataset):
        Xs.append(X.mean(dim=(1, 2)))

    Xs = torch.stack(Xs, dim=0)
    ycc_means = Xs.mean(dim=0)

    print(ycc_means)

    ycc_var = torch.zeros(3)

    for X, _ in tqdm(dataset):
        diff = (ycc_means[:, None, None] - X) ** 2
        ycc_var += diff.mean(dim=(1, 2))

    ycc_var /= len(dataset) - 1

    print(ycc_var)
