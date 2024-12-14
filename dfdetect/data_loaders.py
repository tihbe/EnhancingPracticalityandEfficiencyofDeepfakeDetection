from operator import indexOf
import cv2
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from dfdetect.utils import video_to_frames
import torch
from pathlib import Path
from dfdetect.config import Paths
from torch.utils.data import Subset


class DFDC(Dataset):
    def __init__(self, directory, transforms=None, is_test=False):
        self.desc = None
        self.transforms = transforms
        self.last_fps = 0
        self.last_codec = 0
        self.is_test = is_test

        if is_test:
            self.desc = pd.read_csv(os.path.join(directory, "labels.csv"))
            self.desc = self.desc.rename(columns={"filename": "index"})
            self.desc.insert(len(self.desc.keys()), "path", None)
            self.desc["path"] = self.desc["path"].astype(object)
            self.add_subpath_test(directory)

            for subdirectory in range(5):
                sub_path = os.path.join(directory, str(subdirectory))
                if not os.path.exists(sub_path):
                    continue

                self.add_subpath_test(sub_path)

            self.desc["label"] = self.desc["label"].apply(
                lambda label: "FAKE" if int(label) else "REAL"
            )
        else:
            for path in Path(directory).rglob("metadata.json"):
                self.add_subpath(path.parent)

    def add_subpath_test(self, directory):
        videos = os.listdir(directory)
        sub_mask = self.desc["index"].isin(videos)
        self.desc.loc[sub_mask, "path"] = self.desc["index"][sub_mask].apply(
            lambda filename: os.path.join(directory, filename)
        )

    def label_name(self, label: int):
        return "fake" if label else "real"

    def add_subpath(self, directory):
        new_df = pd.read_json(os.path.join(directory, "metadata.json")).transpose()
        new_df = new_df.reset_index()
        new_df["path"] = new_df["index"].apply(
            lambda filename: os.path.join(directory, filename)
        )

        if self.desc is None:
            self.desc = new_df
        else:
            self.desc = pd.concat((self.desc, new_df))

    def __len__(self) -> int:
        return len(self.desc)

    def __getitem__(self, index: int):
        meta = self.desc.iloc[index]
        path = meta["path"]
        video_rgb_gen, self.last_fps, self.last_codec = video_to_frames(path)
        video_rgb = np.array(list(video_rgb_gen))

        if self.transforms is not None:
            video_rgb = self.transforms(video_rgb)
        return video_rgb, int(meta["label"] == "FAKE")

    def get_filename(self, index: int):
        meta = self.desc.iloc[index]
        return meta["index"]


class DFDC_preprocessed(Dataset):
    """Dataset class assuming the faces were preextracted and stored as video"""

    def __init__(
        self, preprocessed_directory, transforms=None, is_test=False, limit_fps=None
    ):
        self.desc = None
        self.transforms = transforms
        self.limit_fps = limit_fps

        if is_test:
            self.desc = pd.read_csv(os.path.join(preprocessed_directory, "labels.csv"))
            self.desc = self.desc.rename(columns={"filename": "index"})
            self.desc.insert(len(self.desc.keys()), "matchs", None)
            self.desc["matchs"] = self.desc["matchs"].astype("object")
            self.desc["label"] = self.desc["label"].apply(
                lambda label: "FAKE" if int(label) else "REAL"
            )
            files_in_dir = os.listdir(preprocessed_directory)

            for index, row in list(self.desc.iterrows()):
                filename = row["index"]
                extensionless = os.path.splitext(filename)[0]
                matchs = [
                    os.path.join(preprocessed_directory, fn)
                    for fn in files_in_dir
                    if fn.startswith(extensionless)
                ]
                if len(matchs) == 0:
                    self.desc.drop(index, inplace=True)
                else:
                    self.desc.at[index, "matchs"] = matchs

        else:
            for path in Path(preprocessed_directory).rglob("metadata.json"):
                self.add_subpath(path.parent)

            self.labels = list(
                int(self.desc.iloc[i]["label"] == "FAKE") for i in range(len(self.desc))
            )

    def add_subpath(self, directory):
        metadata_path = os.path.join(directory, "metadata.json")
        new_df = pd.read_json(metadata_path).transpose()
        new_df = new_df.reset_index()
        files_in_dir = os.listdir(directory)
        new_df.insert(len(new_df.keys()), "matchs", None)
        new_df["matchs"] = new_df["matchs"].astype("object")

        for index, row in list(new_df.iterrows()):
            filename = row["index"]
            extensionless = os.path.splitext(filename)[0]
            matchs = [
                os.path.join(directory, fn)
                for fn in files_in_dir
                if fn.startswith(extensionless)
            ]
            if len(matchs) == 0:
                new_df.drop(index, inplace=True)
            else:
                new_df.at[index, "matchs"] = matchs

        if self.desc is None:
            self.desc = new_df
        else:
            self.desc = pd.concat((self.desc, new_df))

    def __len__(self) -> int:
        return len(self.desc)

    def __getitem__(self, index: int):
        meta = self.desc.iloc[index]
        matchs = meta["matchs"]
        videos = []
        keys = []
        for m in matchs:
            gen, fps, _ = video_to_frames(m)
            frames = list(gen)
            keys.append(len(frames))

            if self.limit_fps is not None and fps > self.limit_fps:
                frames = frames[:: int(fps / self.limit_fps)]
            videos.append(frames)

        # Sort videos by key
        keys_and_videos = sorted(zip(keys, videos), key=lambda x: x[0])
        videos = list(zip(*keys_and_videos))[1]

        if self.transforms is not None:
            for i, v in enumerate(videos):
                if len(v) == 0:
                    raise Exception(f"Video is corrupted: {matchs[i]}")

            videos = torch.cat(list(self.transforms(v) for v in videos))

        return videos, int(meta["label"] == "FAKE")

    def get_filename(self, index: int):
        meta = self.desc.iloc[index]
        return meta["index"]


class DFDC_preprocessed_single_frames(Dataset):
    def __init__(self, directory, transforms=None):
        self.directory = directory
        self.transforms = transforms
        self.desc = pd.read_csv(os.path.join(directory, "labels.csv"))
        self.labels = (self.desc["label"] == "FAKE").to_numpy()
        self.extra_transforms = None

    def __len__(self) -> int:
        return len(self.desc)

    def __getitem__(self, index: int):
        meta = self.desc.iloc[index]
        try:
            image = cv2.cvtColor(cv2.imread(meta["file_name"]), cv2.COLOR_BGR2RGB)
        except Exception:
            print("Error loading file:", meta["file_name"])
            raise
        if self.transforms is not None:
            image = self.transforms(image)
        return image, int(meta["label"] == "FAKE")

    def label_name(self, label):
        return "fake" if label else "real"


class Oversampled(Dataset):
    # Non-random over sampler to balance the classes
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(self.dataset)))
        if isinstance(dataset, Subset):
            labels = self.dataset.dataset.labels
            labels = np.array(labels)[dataset.indices]
        else:
            labels = self.dataset.labels

        mask_0 = labels == 0
        mask_1 = ~mask_0
        count_0 = np.sum(mask_0)
        count_1 = np.sum(mask_1)

        if count_0 < count_1:  # add to class 0
            missing_samples = count_1 - count_0
            matching_samples = np.flatnonzero(mask_0).tolist()

        elif count_1 < count_0:  # add to class 1
            missing_samples = count_0 - count_1
            matching_samples = np.flatnonzero(mask_1).tolist()

        ratio = missing_samples / len(matching_samples)
        self.indices += (
            int(ratio) * matching_samples
            + matching_samples[: int((ratio - int(ratio)) * len(matching_samples))]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class CelebDFV2(Dataset):
    def __init__(
        self, directory=Paths.CelebDFV2.dataset_path, is_train=True, transforms=None
    ):
        self.directory = directory
        self.transforms = transforms
        self.is_train = is_train
        self.real_directories = ["Celeb-real", "YouTube-real"]
        self.fake_directories = ["Celeb-synthesis"]

        list_directories = lambda dirs: sum(
            (
                [
                    os.path.join(dir, fname)
                    for fname in os.listdir(os.path.join(self.directory, dir))
                ]
                for dir in dirs
            ),
            [],
        )

        self.fake_files = list_directories(self.fake_directories)
        self.real_files = list_directories(self.real_directories)

        self.test_files = pd.read_csv(
            os.path.join(directory, "List_of_testing_videos.txt"), sep=" ", header=None
        )
        self.test_files = set(self.test_files[1].to_list())

        previous_len = len(self.fake_files), len(self.real_files)
        if is_train:
            self.fake_files = list(set(self.fake_files) - self.test_files)
            self.real_files = list(set(self.real_files) - self.test_files)
        else:
            self.fake_files = list(set(self.fake_files).intersection(self.test_files))
            self.real_files = list(set(self.real_files).intersection(self.test_files))

        assert (
            len(self.fake_files) != previous_len[0]
            and len(self.real_files) != previous_len[1]
        )

        self.files = [
            os.path.join(directory, fn) for fn in self.fake_files + self.real_files
        ]
        self.labels = [0] * len(self.fake_files) + [1] * len(self.real_files)

    def label_name(self, label):
        return "real" if label else "fake"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        file_name = self.files[index]
        label = self.labels[index]
        video_rgb_gen, self.last_fps, self.last_codec = video_to_frames(file_name)
        video_rgb = np.array(list(video_rgb_gen))

        if self.transforms is not None:
            video_rgb = self.transforms(video_rgb)

        return video_rgb, label

    def split_train_val(self, ratio=0.8, gen=np.random.default_rng(0x1B)):
        from torch.utils.data import Subset
        import re

        assert self.is_train

        indices = np.arange(len(self.fake_files), dtype=int)
        gen.shuffle(indices)

        cut_off = int(ratio * len(indices))
        train_indices = []
        val_indices = []

        # For each fake file, append original file and fake file together to either train or val

        for i, index in enumerate(indices):
            fake_file = self.fake_files[index]
            directory = os.path.dirname(fake_file)
            filename = os.path.basename(fake_file)
            parts = filename.split("_")
            r = re.compile(f"Celeb-real/{parts[0]}_{parts[2]}.*")
            original_indices = [
                i for i in range(len(self.real_files)) if r.match(self.real_files[i])
            ]

            if i < cut_off:
                train_indices.append(index)
                train_indices += original_indices
            else:
                val_indices.append(index)
                val_indices += original_indices

        unused_indices = list(
            set(range(len(self))) - set(train_indices) - set(val_indices)
        )
        gen.shuffle(unused_indices)
        train_indices += unused_indices[: int(ratio * len(unused_indices))]
        val_indices += unused_indices[int(ratio * len(unused_indices)) :]
        return Subset(self, train_indices), Subset(self, val_indices)


class CelebDFV2_preprocessed(CelebDFV2):
    def __init__(
        self,
        directory=Paths.CelebDFV2.preprocessed_path,
        is_train=True,
        transforms=None,
    ):
        super().__init__(directory, is_train, transforms)

    def __getitem__(self, index: int):
        file_name = self.files[index]
        label = self.labels[index]

        try:
            image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        except Exception:
            print("Error loading file:", file_name)
            raise

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


if __name__ == "__main__":
    from tqdm import tqdm

    from torchvision import transforms
    from dfdetect.data_loaders import DFDC_preprocessed
    from dfdetect.utils import CropResize, FrameBasedTransforms

    all_transforms = FrameBasedTransforms(
        transforms.Compose(
            [
                CropResize(128),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    )
    all_transforms = transforms.Compose([all_transforms])

    dataset = DFDC_preprocessed("./dfdc_preprocessed", transforms=all_transforms)
    total = len(dataset)
    print(f"Total: {total}")
    fake_count = 0
    pbar = tqdm(dataset)
    for i, (X, y) in enumerate(pbar):
        fake_count += y
        pbar.set_postfix(fake_ratio=fake_count / (i + 1))
    print(f"Ratio of fake: {fake_count / total}")
    print(f"Ratio of real: {1 - fake_count / total}")
