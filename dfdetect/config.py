import os
import argparse


class Paths:

    checkpoints = os.environ.get("CHECKPOINT_PATH", "./checkpoints")
    previous_checkpoint = os.environ.get("PREVIOUS_CHECKPOINT", None)

    class DFDC:
        dataset_path = os.environ.get("DFDC_DATASET_PATH", "./dfdc2020")
        train_set = os.path.join(dataset_path, "train_videos")
        test_set = os.path.join("DFDC_DATASET_PATH_TEST", "./dfdc_test_set")
        validation_set = os.path.join("DFDC_DATASET_PATH_VALID", "./validation_set")
        preprocessed_dataset_train = os.environ.get(
            "DFDC_PREPROCESSED_DATASET_PATH",
            "./dfdc_preprocessed",
        )
        preprocessed_dataset_test = os.environ.get(
            "DFDC_PREPROCESSED_DATASET_PATH_TEST",
            "./dfdc_preprocessed_test",
        )

        preprocessed_dataset_single_frames_train = os.environ.get(
            "DFDC_PREPROCESSED_DATASET_PATH_SINGLE_FRAMES",
            "./dfdc_preprocessed_frames",
        )
        preprocessed_dataset_single_frames_val = os.environ.get(
            "DFDC_PREPROCESSED_DATASET_PATH_SINGLE_FRAMES_VAL",
            "./dfdc_preprocessed_frames_validation",
        )
        preprocessed_dataset_single_frames_test = os.environ.get(
            "DFDC_PREPROCESSED_DATASET_PATH_SINGLE_FRAMES_TEST",
            "./dfdc_preprocessed_frames_test",
        )

    class CelebDFV2:
        dataset_path = os.environ.get("CELEB_DFV2_DATASET_PATH", "./celeb-df-v2")
        preprocessed_path = os.environ.get(
            "CELEB_DFV2_PREPROCESSED_PATH",
            "./celeb-df-v2-preprocessed",
        )


class CLA:  # Command Line Arguments
    _instance = None

    def __new__(cls, *args, **kwargs):  # Singleton behavior
        if not cls._instance:
            cls._instance = super(CLA, cls).__new__(cls, *args, **kwargs)
            cls.parser = argparse.ArgumentParser(description="DFDetect library")
        return cls._instance

    @staticmethod
    def register(name, default=None, **kwargs) -> argparse.ArgumentParser:
        if not name.startswith("--"):
            name = "--" + name
        if default is not None:
            value_type = type(default)
            if value_type is bool:
                kwargs["action"] = "store_true"
                CLA().parser.add_argument(name, **kwargs)
                kwargs["action"] = "store_false"
                raw_name = name.replace("--", "")
                kwargs["dest"] = raw_name
                CLA().parser.add_argument(name.replace("--", "--no_"), **kwargs)
                CLA().parser.set_defaults(**{raw_name: default})
                return
            elif value_type is list:
                kwargs["action"] = "append"
                raw_name = name.replace("--", "")
                CLA().parser.set_defaults(**{raw_name: default})
            else:
                kwargs.update({"default": default, "type": value_type})
        CLA().parser.add_argument(name, **kwargs)

    @staticmethod
    def parse():
        for k, v in CLA.to_dict().items():
            setattr(CLA, k, v)

    @staticmethod
    def to_dict():
        return vars(CLA().parser.parse_args())

    @staticmethod
    def to_hash():
        from dfdetect.utils import hash_dict

        return str(abs(hash_dict(CLA.to_dict())))[:20]
