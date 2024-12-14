import math
from enum import IntFlag
from typing import Any

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from dfdetect.models.simple_vit import SimpleViT
from dfdetect.models.srnet import SRNet
from dfdetect.models.tnt import TNT
from dfdetect.utils import dct_2d
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from fairscale.nn import auto_wrap
from madgrad import MADGRAD
from timm.models.senet import SENet
from torch import Tensor, nn
from torchmetrics import AUROC, Accuracy, F1Score, ConfusionMatrix
from torchvision.ops import SqueezeExcitation
from transformers import ViTFeatureExtractor, ViTModel
from collections import namedtuple
from dfdetect.config import CLA
from fastai.layers import SelfAttention

# TODO:
# Split models for feature extraction?
# Video - segment by segment?


StepResults = namedtuple("Result", "preds targets")


class BaseLightningModel(pl.LightningModule):
    """Additional operations for Pytorch-Lightning models specific to image binary classification (deepfake detection)."""

    def __init__(self, decision_threshold=0.5, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.decision_threshold = decision_threshold
        self.acc = Accuracy(threshold=decision_threshold)
        self.f1 = F1Score(threshold=decision_threshold)
        self.auroc = AUROC()
        # self.class_weights = torch.tensor([9.5, 0.5])

    def configure_optimizers(self):
        return MADGRAD(self.parameters())

    def common_step(self, batch, batch_idx, with_preds=False):
        x, y_hat = batch
        # weights = (1 - y_hat) * self.class_weights[0] + y_hat * self.class_weights[1]
        y = self(x).squeeze(dim=-1)
        loss = F.binary_cross_entropy(y, y_hat.float())  # , weight=weights)
        probs = y.detach()
        results = {
            "loss": loss,
            "accuracy": self.acc(probs, y_hat),
            "f1_score": self.f1(probs, y_hat),
            "auroc": self.auroc(probs, y_hat),
        }
        if with_preds:
            preds = torch.round(probs + 0.5 - self.decision_threshold).long()
            return results, StepResults(preds, y_hat)
        return results

    def training_step(self, train_batch, batch_idx):
        metrics = self.common_step(train_batch, batch_idx)
        metrics = {"train_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

        return metrics["train_loss"]

    def validation_step(self, val_batch, batch_idx) -> StepResults:
        metrics, step_results = self.common_step(val_batch, batch_idx, with_preds=True)
        metrics = {"val_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return step_results

    def test_step(self, test_batch, batch_idx) -> StepResults:
        metrics, step_results = self.common_step(test_batch, batch_idx, with_preds=True)
        metrics = {"test_" + k: v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return step_results

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([tmp.preds for tmp in outputs])
        targets = torch.cat([tmp.targets for tmp in outputs])
        try:
            self.loggers[0].experiment.log_confusion_matrix(preds, targets)
        except Exception as e:
            print("Exception occured when logging confusion matrix:", e)

        return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([tmp.preds for tmp in outputs])
        targets = torch.cat([tmp.targets for tmp in outputs])
        try:
            self.loggers[0].experiment.log_confusion_matrix(preds, targets)
        except Exception as e:
            print("Exception occured when logging confusion matrix:", e)
        return super().test_epoch_end(outputs)


class PositionalEncoding(nn.Module):
    # Positional Encoding from PyTorch https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # Converted to use batch first

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Token(nn.Module):
    def __init__(self, features_size):
        super(Token, self).__init__()
        self.token = nn.Parameter(torch.randn(1, 1, features_size))

    def forward(self):
        return self.token


class FeatureType(IntFlag):
    RGB = 1
    YCC = 2
    DCT = 4
    FFT = 8
    SPATIAL = RGB | YCC
    SPECTRAL = DCT | FFT
    ALL = SPATIAL | SPECTRAL


class DFDetectModel(BaseLightningModel):
    class FeatureCombiner(IntFlag):
        TRANSFORMER = 1
        GRU = 2

    def __init__(
        self,
        image_size: int,
        features: FeatureType = FeatureType.RGB | FeatureType.DCT,
        combiner: FeatureCombiner = FeatureCombiner.GRU,
        **kwargs: Any,
    ):
        super(DFDetectModel, self).__init__(**kwargs)

        self.features = features
        self.features_size = 0
        self.combiner = combiner

        if features & FeatureType.RGB:
            patch_dim = 128
            self.rgb_features_extractor = TNT(
                image_size=image_size,  # size of image
                patch_dim=patch_dim,  # dimension of patch token
                pixel_dim=24,  # dimension of pixel token
                patch_size=16,  # patch size
                pixel_size=4,  # pixel size
                depth=4,  # depth
                with_mlp_head=False,  # Remove MLP Head from TnT Model
                attn_dropout=0.1,  # attention dropout
                ff_dropout=0.1,  # feedforward dropout
            )
            self.features_size += patch_dim

        if features & FeatureType.DCT:
            patch_dim = 64
            self.spectral_features_extractor = TNT(
                image_size=image_size,  # size of image
                patch_dim=patch_dim,  # dimension of patch token
                pixel_dim=24,  # dimension of pixel token
                patch_size=8,  # patch size
                pixel_size=2,  # pixel size
                depth=4,  # depth
                heads=2,
                dim_head=64,
                with_mlp_head=False,  # Remove MLP Head from TnT Model
                attn_dropout=0.1,  # attention dropout
                ff_dropout=0.1,  # feedforward dropout
            )
            self.features_size += patch_dim

        if combiner == DFDetectModel.FeatureCombiner.TRANSFORMER:
            self.temporal_features_extractor_layer = nn.TransformerEncoderLayer(
                self.features_size,
                nhead=2,
                dim_feedforward=128,
                batch_first=True,
                norm_first=True,
            )
            self.positional_encoder = PositionalEncoding(self.features_size)
            self.temporal_features_extractor = nn.TransformerEncoder(
                self.temporal_features_extractor_layer, num_layers=2
            )
            self.cls_token = Token(self.features_size)

            self.mlp_head = nn.Linear(self.features_size, 1)
        elif combiner == DFDetectModel.FeatureCombiner.GRU:
            self.rnn_size = 64
            self.rnn = nn.GRU(
                input_size=self.features_size,
                hidden_size=self.rnn_size,
                num_layers=1,
                batch_first=True,
                # dropout=0.1, # All layers except the last, if num_layers > 1
            )

            self.output_layer = nn.Linear(self.rnn_size, 1)

    def configure_sharded_model(self):
        self.rgb_features_extractor = auto_wrap(self.rgb_features_extractor)
        if self.combiner == DFDetectModel.FeatureCombiner.TRANSFORMER:
            self.temporal_features_extractor = auto_wrap(
                self.temporal_features_extractor
            )
            self.mlp_head = auto_wrap(self.mlp_head)
            self.cls_token = auto_wrap(self.cls_token)
        elif self.combiner == DFDetectModel.FeatureCombiner.GRU:
            self.rnn = auto_wrap(self.rnn)
            self.output_layer = auto_wrap(self.output_layer)

    def forward(self, x):
        # Flatten time dimension for image-level feature extraction
        b, t, *_ = x.shape
        x_flat_time = rearrange(x, "b t c h w -> (b t) c h w")

        features = []
        if self.features & FeatureType.RGB:
            y_rgb = self.rgb_features_extractor(x_flat_time)
            features.append(y_rgb)

        if self.features & FeatureType.DCT:
            y_spectral = dct_2d(x_flat_time)
            y_spectral = self.spectral_features_extractor(y_spectral)
            features.append(y_spectral)

        # Combine features
        y = torch.cat(features, dim=1)

        # Unflatten time dimension for temporal feature extraction
        y = rearrange(y, "(b t) f -> b t f", b=b, t=t)

        if self.combiner == DFDetectModel.FeatureCombiner.TRANSFORMER:
            # Add classification token
            cls_tokens = repeat(self.cls_token(), "1 1 d -> b 1 d", b=b)
            y = torch.cat((cls_tokens, y), dim=1)

            # Add positional embedding
            y = self.positional_encoder(y)

            # Temporal transformer
            y = self.temporal_features_extractor(y)

            # Get features from classif token
            y = y[:, 0]  # TODO, use mean pooling ? y.mean(dim=1)

            # Classification
            y = self.mlp_head(y)

        elif self.combiner == DFDetectModel.FeatureCombiner.GRU:
            # Rnn with mean of temporal outputs
            y = self.rnn(y).mean(dim=1)

            # Classification
            y = self.output_layer(y)

        return torch.sigmoid(y)  # As probability


class TNTPl(BaseLightningModel):
    def __init__(self, im_size, **kwargs):
        super().__init__(**kwargs)
        self.model = timm.create_model(
            "tnt_s_patch16_224", pretrained=True
        )  # Pretrained on imagenet
        self.model.head = nn.Linear(self.model.embed_dim, 1)  # Replace mlp head

        # self.model = TNT(
        #     image_size=im_size,
        #     patch_dim=512,  # dimension of patch token
        #     pixel_dim=24,  # dimension of pixel token
        #     patch_size=16,  # patch size
        #     pixel_size=4,  # pixel size
        #     depth=6,  # depth
        #     with_mlp_head=True,  # Remove MLP Head from TnT Model
        #     num_classes=1,
        #     attn_dropout=0.1,  # attention dropout
        #     ff_dropout=0.1,  # feedforward dropout
        # )

    def forward(self, x):
        y = self.model(x)
        return torch.sigmoid(y)  # As probability


import os
from .srnet_pre import Srnet


class SrnetPl(BaseLightningModel):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.model = Srnet(in_channels=in_channels)
        if in_channels == 1 and str(self.device) == "cpu":
            if not os.path.exists("SRNet_model_weights.pt"):
                print("Downloading SRNet weights")
                import requests

                r = requests.get(
                    "https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet/blob/master/checkpoints/SRNet_model_weights.pt?raw=true"
                )
                with open("SRNet_model_weights.pt", "wb") as f:
                    f.write(r.content)

            self.model.load_state_dict(
                torch.load("SRNet_model_weights.pt")["model_state_dict"]
            )
        self.model.fc = nn.Linear(512, 1)

    def forward(self, image):
        y = self.model(image)
        return torch.sigmoid(y)


class TimmModel(BaseLightningModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model = timm.create_model(
            model_name, pretrained=True, in_chans=3, **kwargs
        )  # features_only=True,
        self.model.reset_classifier(num_classes=1)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class SteganalysisModel(BaseLightningModel):
    # Adapted from https://www.kaggle.com/competitions/alaska2-image-steganalysis/discussion/168548
    def __init__(
        self,
        features: FeatureType = FeatureType.YCC | FeatureType.DCT,
        original_stegano=True,
        spatial_model="legacy_seresnet18",
        pretrained=True,
        **kwargs,
    ):
        super(SteganalysisModel, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.feature_size: int = 0
        self.features: FeatureType = features

        if features & FeatureType.YCC:
            self.ycc_feat_extractor: SENet = timm.create_model(
                spatial_model, pretrained=pretrained
            )
            if original_stegano:
                if spatial_model == "legacy_seresnet18":
                    self.ycc_feat_extractor.layer0.conv1.stride = (1, 1)
                    self.ycc_feat_extractor.pool0 = nn.Identity()
                    self.ycc_feat_extractor.last_linear = nn.Identity()
                elif spatial_model == "seresnet50":
                    self.ycc_feat_extractor.conv1.stride = (1, 1)
                    self.ycc_feat_extractor.maxpool = nn.Identity()
                    self.ycc_feat_extractor.fc = nn.Identity()
                else:
                    raise NotImplementedError()

            self.feature_size += self.ycc_feat_extractor.num_features

        if features & FeatureType.DCT:
            dct_patch_size = 8
            self.pre_dct = Rearrange(
                "b c (h p1) (w p2) -> b c h w p1 p2",
                p1=dct_patch_size,
                p2=dct_patch_size,
            )
            self.post_dct = Rearrange("b c h w p1 p2 -> b (c p1 p2) h w")
            self.oh_thresh = 5
            self.dct_conv_size = 192 * (self.oh_thresh + 1)

            self.dct_layers = nn.ModuleList()
            self.SE_blocks = nn.ModuleList()
            for i in range(6):
                self.dct_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.dct_conv_size,
                            self.dct_conv_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.BatchNorm2d(self.dct_conv_size),
                        nn.ReLU(),
                    )
                )
                self.SE_blocks.append(
                    SqueezeExcitation(self.dct_conv_size, self.dct_conv_size)
                )

            self.dct_global_pooling = nn.AdaptiveAvgPool2d(1)
            self.feature_size += self.dct_conv_size

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def dct_onehot(self, x):
        # Idea from http://www.ws.binghamton.edu/fridrich/Research/OneHot_Revised.pdf
        # Adapted from code: https://github.com/YassineYousfi/OneHotConv/blob/bfb4a5d0810d36cafa2a604e3cbd662f079efa44/retriever.py#L47
        T = self.oh_thresh
        x = torch.abs(x).clip_(0, T)
        x_oh = torch.zeros(x.shape + (T + 1,), device=x.device, dtype=x.dtype)

        for ch in range(x.shape[1]):
            x_oh[:, ch] = torch.isclose(
                x[
                    :,
                    ch,
                ].unsqueeze(dim=-1),
                torch.arange(T + 1, device=x.device, dtype=x.dtype),
            )
        return rearrange(x_oh, "b c h w o -> b (c o) h w")

    def forward(self, x):
        extracted_features = []
        if self.features & FeatureType.YCC:
            features = self.ycc_feat_extractor(x)
            extracted_features.append(features)

        if self.features & FeatureType.DCT:
            patched_x = self.pre_dct(x)
            spectral_components = dct_2d(patched_x)
            spectral_components = self.post_dct(spectral_components)
            features = self.dct_onehot(spectral_components)
            for i in range(6):
                features = self.SE_blocks[i](self.dct_layers[i](features)) + features
            features = self.dct_global_pooling(features).flatten(start_dim=1)
            extracted_features.append(features)

        extracted_features = torch.cat(extracted_features, dim=1)
        return self.classifier(extracted_features)


from fastai.layers import SelfAttention


class SRNetDouble(BaseLightningModel):
    def __init__(
        self,
        features: FeatureType = FeatureType.YCC | FeatureType.DCT,
        srnet_args_spatial={},
        srnet_args_spectral={},
        with_attention=False,
        single_layer_fc=True,
        **kwargs,
    ):
        super(SRNetDouble, self).__init__(**kwargs)

        self.save_hyperparameters()

        self.feature_size: int = 0
        self.features: FeatureType = features

        if features & FeatureType.SPATIAL:
            self.spatial_feat_extractor: SRNet = SRNet(
                in_channels=3, **srnet_args_spatial
            )
            self.feature_size += self.spatial_feat_extractor.nb_features

        if features & FeatureType.SPECTRAL:
            self.spectral_feat_extractor: SRNet = SRNet(
                in_channels=3, **srnet_args_spectral
            )
            self.feature_size += self.spectral_feat_extractor.nb_features

            if features & FeatureType.DCT:
                dct_patch_size = srnet_args_spectral.get("dct_patch_size", 8)
                self.pre_dct = Rearrange(
                    "b c (h p1) (w p2) -> b c h w p1 p2",
                    p1=dct_patch_size,
                    p2=dct_patch_size,
                )
                self.post_dct = Rearrange("b c h w p1 p2 -> b c (h p1) (w p2)")

        self.with_attention = with_attention
        if with_attention:
            assert features & FeatureType.SPATIAL
            assert features & FeatureType.SPECTRAL
            assert (
                self.spatial_feat_extractor.nb_features
                == self.spectral_feat_extractor.nb_features
            )
            nb_features = self.spatial_feat_extractor.nb_features
            self.att_spectral = SelfAttention(nb_features)
            self.att_spatial = SelfAttention(nb_features)

        if single_layer_fc:
            self.classifier = nn.Sequential(
                # nn.Linear(self.feature_size, self.feature_size // 2),
                # nn.Dropout(0.3),
                # nn.ReLU(),
                nn.Linear(self.feature_size, 1),
                nn.Sigmoid(),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size // 2),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(self.feature_size // 2, 1),
                nn.Sigmoid(),
            )

    def load_state_dict(self, state_dict, strict=True):
        """Function to rename some keys in the state dict of older checkpoints before loading"""
        from collections import OrderedDict

        rename_dict = {
            "ycc_feat_extractor": "spatial_feat_extractor",
            "dct_feat_extractor": "spectral_feat_extractor",
            "bn": "norm",
        }

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            for old, new in rename_dict.items():
                if old in key:
                    key = key.replace(old, new)

            new_state_dict[key] = value

        return super().load_state_dict(new_state_dict, strict)

    def forward(self, x):
        extracted_features = []
        if self.features & FeatureType.SPATIAL:
            features = self.spatial_feat_extractor.feat_extractor(x)
            extracted_features.append(features)

        if self.features & FeatureType.SPECTRAL:
            if self.features & FeatureType.DCT:
                patched_x = self.pre_dct(x)
                spectral_components = dct_2d(patched_x)
                spectral_components = self.post_dct(spectral_components)
                # features = self.dct_onehot(spectral_components)
            elif self.features & FeatureType.FFT:
                spectral_components = torch.fft.fft2(x)
                spectral_components = F.relu(
                    torch.log(spectral_components.abs() + 1e-6)
                )

            features = self.spectral_feat_extractor.feat_extractor(spectral_components)
            extracted_features.append(features)

        if self.with_attention:
            extracted_features = [
                extracted_features[0] + self.att_spatial(extracted_features[0]),
                extracted_features[1] + self.att_spectral(extracted_features[1]),
            ]

        # Global pooling
        extracted_features_pooled = []
        for feat in extracted_features:
            feat = torch.mean(feat, dim=(2, 3), keepdim=True)
            feat = feat.view(feat.size(0), -1)
            extracted_features_pooled.append(feat)

        extracted_features_pooled = torch.cat(extracted_features_pooled, dim=1)
        return self.classifier(extracted_features_pooled)


# # google/vit-base-patch16-224-in21k
# class Model(BaseLightningModel):
#     def __init__(
#         self,
#         rnn_size=128,
#         with_rgb_feature_extractor=True,
#         with_spectral_feature_extractor=False,
#         use_last_rnn_state=True,
#         reduc_size=128,
#     ):
#         super(Model, self).__init__()

#         self.feature_size = 0
#         self.with_rgb_feature_extractor = with_rgb_feature_extractor
#         self.with_spectral_feature_extractor = with_spectral_feature_extractor
#         if with_rgb_feature_extractor:
#             # "facebook/data2vec-vision-base",
#             self.rgb_feature_extractor = ViTModel.from_pretrained(
#                 "google/vit-base-patch16-224-in21k", add_pooling_layer=True
#             )  # https://arxiv.org/abs/2010.11929
#             self.rgb_feature_extractor.trainable = False
#             for p in self.rgb_feature_extractor.parameters():
#                 p.requires_grad = False

#             self.feature_size += self.rgb_feature_extractor.config.hidden_size

#         if with_spectral_feature_extractor:
#             self.simple_vit = SimpleViT(
#                 image_size=(224, 224),
#                 patch_size=(16, 16),
#                 num_classes=2,  # Not important, MLP head was removed
#                 dim=64,
#                 depth=4,
#                 heads=2,
#                 mlp_dim=64,
#                 channels=2,
#                 dim_head=64,
#             )
#             self.feature_size += 64
#             self.feat_norm = nn.LayerNorm(224)

#         self.reduc = nn.Linear(self.feature_size, reduc_size)
#         self.rnn_size = rnn_size
#         self.rnn = nn.GRU(
#             input_size=reduc_size,
#             hidden_size=rnn_size,
#             num_layers=1,
#             # batch_first=True,
#             # dropout=0.1, # All layers except the last, if num_layers > 1
#         )

#         self.output_layer = nn.Linear(rnn_size, 1)
#         self.use_last_rnn_state = use_last_rnn_state

#     def forward(self, x):
#         x = torch.swapaxes(x, 0, 1)  # time, batch, features*

#         temporal_batch = []
#         for x_t in x:
#             if self.with_rgb_feature_extractor:
#                 with torch.no_grad():
#                     rgb_features = self.rgb_feature_extractor(x_t).pooler_output.detach()
#             if True or self.with_spectral_feature_extractor:
#                 spectral_components = torch.from_numpy(dctn(x_t.mean(dim=1).cpu().numpy(), s=(64, 64), axes=(1, 2))).to(
#                     x_t.device
#                 )
#                 exit(0)
#                 # spectral_components = dct.dct_2d(x_t.mean(dim=1))
#                 # spectral_components = torch.fft.fft2()
#                 # phase = torch.angle(spectral_components)
#                 # amplitude = torch.abs(spectral_components)
#                 amplitude = self.feat_norm(amplitude)
#                 x_t = torch.stack([amplitude, phase], dim=1)
#                 x_t = self.simple_vit(x_t)
#             x_t = self.reduc(rgb_features)
#             temporal_batch.append(x_t)
#         temporal_batch = torch.stack(temporal_batch, dim=0)

#         x_t, _ = self.rnn(temporal_batch)
#         if self.use_last_rnn_state:
#             x_t = x_t[-1]
#         else:
#             x_t = x_t.mean(dim=0)
#         x_t = self.output_layer(x_t)
#         return torch.sigmoid(x_t)
