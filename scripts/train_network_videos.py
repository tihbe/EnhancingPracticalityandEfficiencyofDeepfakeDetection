import os
import numpy as np
import pytorch_lightning as pl
import torch
from dfdetect.config import Paths
from dfdetect.data_loaders import DFDC_preprocessed
from dfdetect.models.dfdetect_model import DFDetectModel
from dfdetect.utils import CropResize, FrameBasedTransforms, collate_zero_pad
from torch.utils.data import DataLoader
from torchvision import transforms
import dfdetect.utils as utils


def main(
    project_name="DFDC-full",
    seed=0x1B,
    test=False,
    debug=False,
    use_rnn=True,
    # use_last_rnn_state=True,
    # rgb_feature_extractor=True,
    # spectral_feature_extractor=False,
):
    params = locals()
    exp_id = str(utils.hash_dict(params))
    loggers = utils.get_pl_loggers(exp_id, params)
    exp_path = os.path.join(Paths.checkpoints, exp_id)
    pl.seed_everything(seed)
    target_image_size = 128

    all_transforms = FrameBasedTransforms(
        transforms.Compose(
            [
                CropResize(target_image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    )
    all_transforms = transforms.Compose([all_transforms])

    num_workers = 2  # multiprocessing.cpu_count() // 2
    batch_size = 1  # 2 if debug else 2
    accumulate_grad_batches = 32  # 12

    data_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_zero_pad,
        "pin_memory": True,
    }
    if not test:
        data_loader_args["shuffle"] = True

    # model = Model(
    #         use_last_rnn_state=use_last_rnn_state,
    #         with_rgb_feature_extractor=rgb_feature_extractor,
    #         with_spectral_feature_extractor=spectral_feature_extractor,
    #     )
    feature_combiner = (
        DFDetectModel.FeatureCombiner.GRU
        if use_rnn
        else DFDetectModel.FeatureCombiner.TRANSFORMER
    )
    model = DFDetectModel(image_size=target_image_size, combiner=feature_combiner)

    callbacks = []
    if not test:
        # callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss_epoch", patience=10, mode="min"))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=exp_path,
            auto_insert_metric_name=True,
            filename="checkpoint_{epoch:02d}-{val_accuracy_epoch:02.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gpus=1 if test else -1,
        devices=1 if test else -1,
        auto_select_gpus=True,
        max_epochs=1 if debug else 100,
        max_steps=5 if debug else 50_000,
        logger=loggers,
        log_every_n_steps=1,
        val_check_interval=1.0,
        limit_train_batches=1000,
        limit_val_batches=100,  # Small set of validation
        num_sanity_val_steps=1,  # Validate before starting training
        accumulate_grad_batches=accumulate_grad_batches,  # Accumulate gradient for n batches
        default_root_dir=exp_path,
        strategy="ddp_sharded",  # ddp
        detect_anomaly=True,
        callbacks=callbacks,
        profiler="advanced" if debug else None,
    )

    if test:
        test_set = DFDC_preprocessed(
            Paths.DFDC.preprocessed_dataset_test,
            transforms=all_transforms,
            is_test=True,
            limit_fps=2,
        )
        test_loader = DataLoader(
            test_set,
            **data_loader_args,
        )
        trainer.test(model, test_loader, ckpt_path=Paths.previous_checkpoint)
    else:
        dataset = DFDC_preprocessed(
            Paths.DFDC.preprocessed_dataset_train,
            transforms=all_transforms,
            limit_fps=2,
        )
        train_length = int(0.8 * len(dataset))
        (
            train_set,
            val_set,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, len(dataset) - train_length],
            generator=torch.Generator().manual_seed(0x1B),
        )
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
            if not debug:
                trainer.save_checkpoint(os.path.join(exp_path, "final_checkpoint.ckpt"))
        trainer.validate(model, val_loader)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
