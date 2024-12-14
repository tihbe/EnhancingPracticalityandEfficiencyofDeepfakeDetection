# Transforms adapted from https://github.com/selimsef/dfdc_deepfake_challenge
import albumentations as A
import cv2


def isotropically_resize_image(
    img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC
):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(A.DualTransform):
    def __init__(
        self,
        max_side,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        always_apply=False,
        p=1,
    ):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(
        self,
        img,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        **params
    ):
        return isotropically_resize_image(
            img,
            size=self.max_side,
            interpolation_down=interpolation_down,
            interpolation_up=interpolation_up,
        )

    def apply_to_mask(self, img, **params):
        return self.apply(
            img,
            interpolation_down=cv2.INTER_NEAREST,
            interpolation_up=cv2.INTER_NEAREST,
            **params
        )

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class DataAugmentations:
    def __init__(self, size=128):
        self.transforms = A.Compose(
            [
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.05),
                A.HorizontalFlip(),
                A.OneOf(
                    [
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC,
                        ),
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=1,
                ),
                A.PadIfNeeded(
                    min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.7,
                ),
                A.ToGray(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
            ]
        )

    def __call__(self, image):
        return self.transforms(image=image)["image"]
