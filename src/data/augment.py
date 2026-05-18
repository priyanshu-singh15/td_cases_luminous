"""Phone-camera aggressive augmentation for Roboflow/YOLO training."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def phone_camera_augment(train: bool = True) -> A.Compose:
    if not train:
        return A.Compose([A.Resize(640, 640), ToTensorV2()])

    return A.Compose(
        [
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(640, 640, border_mode=0),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.45,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=30, p=0.65),
            A.RandomGamma(gamma_limit=(70, 130), p=0.35),
            A.Perspective(scale=(0.02, 0.06), p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.25,
                rotate_limit=15,
                border_mode=0,
                p=0.55,
            ),
            A.CoarseDropout(
                max_holes=6,
                max_height=48,
                max_width=48,
                fill_value=0,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.25),
    )
