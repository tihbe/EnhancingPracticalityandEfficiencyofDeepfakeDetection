import os

import comet_ml  # Set up hooks in other libraries
import numpy as np
import pytorch_lightning as pl
import torch
from dfdetect.config import Paths, CLA
from dfdetect.data_loaders import (
    DFDC_preprocessed_single_frames,
    Oversampled,
    DFDC,
    CelebDFV2_preprocessed,
    CelebDFV2,
)
from dfdetect.models.dfdetect_model import (
    FeatureType,
    SteganalysisModel,
    TimmModel,
    TNTPl,
    SrnetPl,
    SRNetDouble,
)
import torch.nn.functional as F
from dfdetect.utils import CropResize, Slurm
import dfdetect.utils as utils

from torch.utils.data import DataLoader
from torchvision import transforms
from dfdetect.data_augmentation import DataAugmentations
import dfdetect.preprocessing.face_detection as fd
from copy import copy


def setup_args():
    """Setup all the command line arguments supported in this script."""
    CLA.register("project_name", "DFDC-full-single-images")
    CLA.register("dataset", "DFDC", choices=["DFDC", "celebdfv2"])
    CLA.register("seed", 0x1B)
    CLA.register("test", False)
    CLA.register("valid", False)
    CLA.register("test_all_frames", False)
    CLA.register("debug", False)
    CLA.register("model", "stegano")
    CLA.register("stegano_spatial_model", "legacy_seresnet18")
    CLA.register("target_image_size", 128)
    CLA.register("spatial_features", True)
    CLA.register("spectral_features", True)
    CLA.register("dct_features", True)
    CLA.register("rgb_to_ycc", True)
    CLA.register("rgb_and_ycc", False)
    CLA.register("rgb_to_gray", False)
    CLA.register("batch_size", 64)
    CLA.register("accumulate_grad_batches", 1)
    CLA.register("decision_threshold", 0.5)  # When predicting accuracy and f1 score
    CLA.register("nb_epochs", 200)
    CLA.register("act_function", "relu", choices=["relu", "gelu"])
    CLA.register("srnet_double_nb_features", 512)
    CLA.register("srnet_double_num_type2_layers", 5)
    CLA.register("srnet_double_type_3_layer_sizes", [16, 64, 128, 256])
    CLA.register("srnet_double_type_2_layer_feat_size", 16)
    CLA.register("srnet_double_type_1_kernel_size_spatial", 3)
    CLA.register("srnet_double_type_1_kernel_size_spectral", 2)
    CLA.register("srnet_double_with_attention", False)
    CLA.register("plot_ttest", False)
    CLA.register("confidence_to_stop", 0.01)

    CLA.parse()


def get_transforms(set_type):
    """Get transforms for a single RGB frame, only set_type=train will add probabilistic data augmentations"""
    # Means and variance computed from the training set with the function
    # utils.compute_running_stats(train_set)

    assert not (
        CLA.rgb_to_ycc & CLA.rgb_to_gray
    ), "Only one of rgb_to_ycc and rgb_to_gray can be True"
    assert not (CLA.rgb_and_ycc & CLA.rgb_to_ycc)

    if CLA.rgb_to_ycc:
        means = torch.tensor([0.3443, 0.5621, 0.4715])
        stds = torch.tensor([0.0377, 0.0017, 0.0010]).sqrt_()
    elif CLA.rgb_to_gray:
        means = torch.tensor([0.5])
        stds = torch.tensor([0.0833]).sqrt_()
    else:
        means = torch.tensor([0.485, 0.456, 0.406])
        stds = torch.tensor([0.229, 0.224, 0.225])
    all_transforms = [
        CropResize(CLA.target_image_size),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=means, std=stds),
    ]
    if CLA.rgb_to_ycc:
        all_transforms.insert(0, utils.rgb_to_ycc)
    elif CLA.rgb_to_gray:
        all_transforms.insert(0, utils.rgb_to_gray)
    if set_type == "train":  # Add data augmentation for train set in RGB domain
        all_transforms.insert(0, DataAugmentations())

    return transforms.Compose(all_transforms)


def get_model():

    model_name = CLA.model.lower()
    features = 0
    if CLA.spectral_features:
        features |= FeatureType.DCT if CLA.dct_features else FeatureType.FFT
    if CLA.spatial_features:
        features |= FeatureType.YCC if CLA.rgb_to_ycc else FeatureType.RGB

    if model_name == "stegano":
        model_class = SteganalysisModel
        model_args = dict(
            features=features,
            original_stegano=True,
            spatial_model=CLA.stegano_spatial_model,
        )
    elif model_name == "tnt":
        assert CLA.target_image_size == 224
        model_class = TNTPl
        model_args = dict(im_size=CLA.target_image_size)
    elif model_name == "seresnet":
        model_class = SteganalysisModel
        model_args = dict(
            features=features,
            original_stegano=False,
            spatial_model=CLA.stegano_spatial_model,
        )
    elif model_name == "srnet":
        model_class = SrnetPl
        model_args = dict(in_channels=1 if CLA.rgb_to_gray else 3)
    elif model_name == "srnetdouble":
        model_class = SRNetDouble
        model_args = dict(
            features=features,
            srnet_args_spatial={
                "act_function": F.gelu if CLA.act_function == "gelu" else F.relu,
                "nb_features": CLA.srnet_double_nb_features,
                "type_3_layer_sizes": [
                    int(tmp) for tmp in CLA.srnet_double_type_3_layer_sizes
                ],
                "num_type2_layers": CLA.srnet_double_num_type2_layers,
                "type_2_layer_feat_size": CLA.srnet_double_type_2_layer_feat_size,
                "type_1_kernel_size": CLA.srnet_double_type_1_kernel_size_spatial,
            },
            srnet_args_spectral={
                "act_function": F.gelu if CLA.act_function == "gelu" else F.relu,
                "nb_features": CLA.srnet_double_nb_features,
                "type_3_layer_sizes": [
                    int(tmp) for tmp in CLA.srnet_double_type_3_layer_sizes
                ],
                "num_type2_layers": CLA.srnet_double_num_type2_layers,
                "type_2_layer_feat_size": CLA.srnet_double_type_2_layer_feat_size,
                "type_1_kernel_size": CLA.srnet_double_type_1_kernel_size_spectral,
            },
            with_attention=CLA.srnet_double_with_attention,
        )
    else:
        model_class = TimmModel
        model_args = dict(model_name=model_name)

    model_args["decision_threshold"] = CLA.decision_threshold

    if Paths.previous_checkpoint is not None and os.path.exists(
        Paths.previous_checkpoint
    ):
        model = model_class.load_from_checkpoint(
            Paths.previous_checkpoint, **model_args
        )
    else:
        model = model_class(**model_args)

    return model


def get_dataset():
    if CLA.dataset == "DFDC":
        if CLA.test_all_frames:
            dataset = DFDC(
                Paths.DFDC.test_set if CLA.test else Paths.DFDC.validation_set,
                is_test=True,
            )
            return dataset
        elif CLA.test or CLA.valid:
            dataset = DFDC_preprocessed_single_frames(
                (
                    Paths.DFDC.preprocessed_dataset_single_frames_test
                    if CLA.test
                    else Paths.DFDC.preprocessed_dataset_single_frames_val
                ),
                transforms=get_transforms("test"),
            )
            return dataset
        else:  # training
            train_set = DFDC_preprocessed_single_frames(
                Paths.DFDC.preprocessed_dataset_single_frames_train,
                transforms=get_transforms("train"),
            )

            train_set = Oversampled(
                train_set
            )  # balance classes for training with oversampling

            val_set = DFDC_preprocessed_single_frames(
                Paths.DFDC.preprocessed_dataset_single_frames_val,
                transforms=get_transforms("val"),
            )
            return train_set, val_set
    elif CLA.dataset == "celebdfv2":
        cls, transforms, path = None, None, None
        if CLA.test_all_frames:
            cls = CelebDFV2
            path = Paths.CelebDFV2.dataset_path
        else:
            cls = CelebDFV2_preprocessed
            path = Paths.CelebDFV2.preprocessed_path
            transforms = get_transforms("test")

        dataset = cls(path, is_train=not CLA.test, transforms=transforms)

        if not CLA.test:
            train_set, val_set = dataset.split_train_val(ratio=0.8)
            # train_set, val_set = torch.utils.data.random_split(
            #     dataset, (train_len, len(dataset) - train_len), generator=torch.Generator().manual_seed(CLA.seed)
            # )
            train_set.dataset = copy(dataset)  # Full copy for training transforms
            train_set.dataset.transforms = get_transforms("train")
            if CLA.valid:
                return val_set
            else:
                return Oversampled(train_set), val_set

        return dataset
    return None


def ttest_figure(ttest_dicts):
    samples = list(ttest_dicts.values())
    if len(samples) != 6:
        return False

    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.palettes.color_palette("muted")

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), tight_layout=True, sharey=True)
    axs = axs.flatten()

    for i in range(6):
        ttest_dict = samples[i]
        ax = axs[i]

        frame_means = np.array(
            [np.mean(frame_arr) for frame_arr in ttest_dict["frame_labels"]]
        )

        ax.plot(
            frame_means, ".-", label="Average probability of frame", color=colors[0]
        )
        if "early_stop" in ttest_dict:
            ax.vlines(
                ttest_dict["early_stop"],
                ymin=0,
                ymax=1,
                label="Early stop iteration",
                color=colors[1],
            )
            ax.annotate(
                f"Predicted: {ttest_dict['predicted_label']:.2f}",
                arrowprops=dict(arrowstyle="->"),
                xytext=(ttest_dict["early_stop"] + 0.22 * len(frame_means), 0.6),
                xy=(ttest_dict["early_stop"], 0.5),
                ha="center",
                va="center",
            )

        ax.hlines(
            y=ttest_dict["actual_label"],
            xmin=0,
            xmax=len(frame_means),
            label="Actual label",
            color=colors[2],
        )
        ax.set_xlabel("1 out of 10 frames")
        if i % 3 == 0:
            ax.set_ylabel("Probability")

        ax.set_ylim([-0.05, 1.05])

    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        labelspacing=0.0,
        fancybox=True,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.savefig("ttest_viz.png", bbox_inches="tight")
    fig.savefig("ttest_viz.eps", bbox_inches="tight")
    plt.show()

    return True


def frame_by_frame_loop(trainer, model, loggers, plot_ttest=False):
    """Inference loop to go through a video frame by frame and compute the probability of it being fake."""
    from tqdm.auto import tqdm

    from torchmetrics.functional import accuracy, f1_score, auroc
    from scipy.stats import ttest_1samp

    from collections import defaultdict

    plot_dict = defaultdict(dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_transforms = get_transforms("test")

    dataset = get_dataset()
    sample_order = np.arange(len(dataset))
    if plot_ttest and not CLA.test:
        dataset = Oversampled(dataset)
        sample_order = np.random.RandomState(0x1B).permutation(len(dataset))

    predicted_labels = []
    actual_labels = []

    pbar = tqdm(sample_order)
    with torch.no_grad():
        for sample_id in pbar:
            (frames, label) = dataset[sample_id]
            if len(frames) == 0:
                continue
            actual_labels.append(label)
            frame_labels = []
            predicted_label = None

            all_bboxes = []
            tracked_face_probs = defaultdict(list)

            prediction_done = False
            for frame_nb, frame in enumerate(frames[::10]):
                bboxes = fd.waterfall(frame)
                if len(bboxes) == 0:
                    print("Frame could not be processed")
                    continue
                all_bboxes.append(bboxes)
                tracked_faces = fd.face_tracking(all_bboxes)
                for face_id, face_bboxes in tracked_faces.items():
                    if len(tracked_face_probs[face_id]) != len(face_bboxes):
                        last_bbox = face_bboxes[-1]
                        if last_bbox is None:
                            tracked_face_probs[face_id].append(None)
                        else:
                            croped_frame = utils.crop_frames([frame], [last_bbox])
                            croped_frame = all_transforms(list(croped_frame)[0])
                            preds = model(croped_frame.unsqueeze(0).to(device))
                            tracked_face_probs[face_id].append(
                                preds.squeeze().cpu().numpy()
                            )

                # For each face, we compute the probablity of it being fake based on previous frame probabilities
                face_labels = []
                for face_id, face_probs in tracked_face_probs.items():
                    face_probs = [prob for prob in face_probs if prob is not None]
                    res = ttest_1samp(face_probs, 0.5)  # , alternative="greater")
                    if res.pvalue < CLA.confidence_to_stop:
                        predicted_face_label = (
                            1.0 - res.pvalue if res.statistic >= 0 else 0.0 + res.pvalue
                        )
                        face_labels.append(predicted_face_label)

                # For every face probability with a sufficiently low p-value
                # If any of the face is fake, use that probability as the prediction

                for face_label in face_labels:
                    acd = dataset
                    while isinstance(acd, Oversampled) or isinstance(
                        acd, torch.utils.data.Subset
                    ):
                        acd = acd.dataset
                    if acd.label_name(np.round(face_label).astype(int)) == "fake":
                        predicted_label = face_label
                        prediction_done = True
                        break
                else:
                    # If no face is fake, use the average of the face probabilities
                    if len(face_labels) > 0:
                        predicted_label = np.mean(face_labels)
                        frame_labels.append(predicted_label)
                        prediction_done = True

                if prediction_done:
                    frame_labels.append(predicted_label)
                    if plot_ttest:
                        if "early_stop" not in plot_dict[sample_id]:
                            plot_dict[sample_id]["early_stop"] = frame_nb
                            plot_dict[sample_id]["predicted_label"] = predicted_label
                    else:
                        break

            if len(all_bboxes) == 0:  # If we never found any face in the video
                # Last attempt, crop middle 100x100 from center of first frame
                # And use model output as probability
                frame = frames[0]
                h, w, c = frame.shape
                sh, sw = h // 2 - 50, w // 2 - 50
                face = frame[sh : sh + 100, sw : sw + 100, :]
                predicted_label = model(
                    torch.unsqueeze(all_transforms(face).to(device), dim=0)
                )
                predicted_label = predicted_label.cpu().numpy().flatten()

            predicted_label = None
            if (
                predicted_label is None
            ):  # The confidence was never reached yet we have bboxes
                all_flat_probs = [
                    x
                    for x in sum(list(tracked_face_probs.values()), [])
                    if x is not None
                ]
                res = ttest_1samp(all_flat_probs, 0.5)  # , alternative="greater")
                predicted_label = (
                    1.0 - res.pvalue if res.statistic >= 0 else 0.0 + res.pvalue
                )

            if plot_ttest:
                plot_dict[sample_id]["actual_label"] = label
                plot_dict[sample_id]["frame_labels"] = frame_labels
                if ttest_figure(plot_dict):
                    return

            predicted_labels.append(predicted_label)

            plt = torch.tensor(predicted_labels)
            alt = torch.tensor(actual_labels)
            try:
                pbar.set_postfix(
                    {
                        "acc": accuracy(plt, alt),
                        "f1": f1_score(plt, alt),
                        "roc": auroc(plt, alt),
                    }
                )
            except:
                pass

        prepend = "test_" if CLA.test else "valid_"
        scores = {
            prepend + "acc": accuracy(plt, alt),
            prepend + "f1": f1_score(plt, alt),
            prepend + "roc": auroc(plt, alt),
        }
        print("Final scores:")
        print(scores)

        try:
            loggers[0].experiment.log_confusion_matrix(alt, plt)
        except Exception as e:
            print("Exception occured when logging confusion matrix:", e)

        torch.save(plt, "predicted_labels.pt")
        torch.save(alt, "actual_labels.pt")

        # model.log_dict(scores) # Doesn't work without pl loop
        return scores


def main():
    setup_args()
    exp_id = str(Slurm.job_id() if Slurm.is_active() else CLA.to_hash())
    loggers = utils.get_pl_loggers(exp_id, CLA.to_dict())
    exp_path = os.path.join(Paths.checkpoints, exp_id)
    is_training = not (CLA.test or CLA.valid)

    pl.seed_everything(CLA.seed)

    model = get_model()

    num_workers = Slurm.cpu_count()

    data_loader_args = {
        "batch_size": CLA.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "shuffle": is_training,
    }

    callbacks = []
    if is_training and not CLA.debug:
        # callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss_epoch", patience=10, mode="min"))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=exp_path,
            auto_insert_metric_name=True,
            filename="checkpoint_{epoch:02d}-{val_accuracy_epoch:02.5f}-{val_auroc_epoch:02.5f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator="auto",
        gpus=Slurm.gpu_count() if is_training else 1,
        max_epochs=1 if CLA.debug else CLA.nb_epochs,
        logger=loggers,
        log_every_n_steps=100,
        val_check_interval=1.0,
        num_sanity_val_steps=1,
        accumulate_grad_batches=CLA.accumulate_grad_batches,  # Accumulate gradient for n batches
        default_root_dir=exp_path,
        strategy="ddp",
        detect_anomaly=True,
        callbacks=callbacks,
        profiler="advanced" if CLA.debug else None,
        deterministic=False if is_training else True,
        # gradient_clip_val=0.8,
    )

    if CLA.test or CLA.valid:
        if CLA.test_all_frames:
            frame_by_frame_loop(trainer, model, loggers, CLA.plot_ttest)
        else:
            dataset = get_dataset()

            data_loader = DataLoader(
                dataset,
                **data_loader_args,
            )
            if CLA.test:
                trainer.test(model, data_loader, ckpt_path=Paths.previous_checkpoint)
            else:
                trainer.validate(
                    model, data_loader, ckpt_path=Paths.previous_checkpoint
                )
    else:
        train_set, val_set = get_dataset()

        train_loader = DataLoader(
            train_set,
            **data_loader_args,
        )

        val_loader = DataLoader(
            val_set,
            **data_loader_args,
        )
        try:
            trainer.fit(
                model, train_loader, val_loader, ckpt_path=Paths.previous_checkpoint
            )
        finally:  # After training or on exception, save checkpoint
            if not CLA.debug:
                trainer.save_checkpoint(os.path.join(exp_path, "final_checkpoint.ckpt"))
            if not (CLA.debug or CLA.test or CLA.valid):
                print(f"Best model path: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
