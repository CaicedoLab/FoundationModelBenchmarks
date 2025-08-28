# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence, List, Tuple, Optional, Tuple, Union

import torch
import numpy as np
import numbers
import math
import warnings
from torchvision import transforms
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode, get_dimensions
from torch import nn


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    # return transforms.Normalize(mean=0.449, std=0.226)
    return self_normalize()


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=None)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation, antialias=None),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def make_classification_combined_set(
    *,
    crop_size: int = 128,
) -> transforms.Compose:
    transforms_list = [
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        self_normalize()
    ]
    return transforms.Compose(transforms_list)


class InferenceJUMPFullImage(object):
    def __init__(self):
        self.normalize = transforms.Compose(
            [
                self_normalize(),
            ]
        )
        self.grid = ImageGridInference()
    
    def __call__(self, image, mask):
        crops = self.grid(image, mask)
        if crops is not None:
            crops = [
                self.normalize(crop) for crop in crops 
            ]
            return crops, len(crops)
        else:
            return torch.zeros((224,224)), 0


class ImageGridInference(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def pad(self, img, mask):
        _, height, width = get_dimensions(img)
        padding_tb = (224 - (height % 224)) // 2
        padding_lr = (224 - (width % 224)) // 2
        assert (height + (padding_tb * 2)) % 224 == 0
        assert (width + (padding_lr * 2)) % 224 == 0
        img = transforms.functional.pad(img, padding=(padding_lr, padding_tb), fill=0, padding_mode='constant')
        mask = transforms.functional.pad(mask, padding=(padding_lr, padding_tb), fill=0, padding_mode='constant')
        return img, mask

    def create_grid(self, img, mask):
        _, height, width = get_dimensions(img)

        crops = []
        for i in range(0, height-1, 224):
            for j in range(0, width-1, 224):
                if torch.count_nonzero(transforms.functional.crop(mask, i, j, 224, 224)) > int((224*224) * 0.01):
                    crops.append(transforms.functional.crop(img, i, j, 224, 224))

        if len(crops) != 0:
            return torch.stack(crops)
        else:
            return None
    
    def forward(self, img, mask):
        img, mask = self.pad(img, mask)
        crops = self.create_grid(img, mask)
        return crops


class GuidedRandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.
       Repeats RandomResizedCrop from transforms, but verify that 
       non-empty crop is sampled. Don't fall back to central crop. 
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: torch.Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def guided_precrop(self, img, mask):
        _, height, width = transforms.functional.get_dimensions(img)
        while True:
            i = torch.randint(0, height + 1, size=(1,)).item()
            j = torch.randint(0, width + 1, size=(1,)).item()
            if torch.count_nonzero(transforms.functional.crop(mask, i, j, self.size[0], self.size[1])) > int((224*224) * 0.01):
                return transforms.functional.crop(img, i, j, self.size[0], self.size[1])
            else: continue

    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img = self.guided_precrop(img, mask)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string


# The code of implementation augmentations (all code below) is copied from
# Doron et al. "Unbiased single-cell morphology with self-supervised vision transformers".

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.InterpolationMode("bilinear"),
    ):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: torch.Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        #         width, height = torchvision.transforms.functional._get_image_size(img)
        width, height = img.shape[1:]
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )


class Change_contrast(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = transforms.functional.adjust_contrast(
                img[ind][None, ...], factor
            )
        return img


class Intensity_shift_JUMP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = np.random.uniform(-0.3, 0.3)
            img[ind] = np.clip(img[ind] + factor, 0, 1)
        return img


class Change_brightness_JUMP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = np.random.uniform(0.5, 1.5)
            img[ind] = transforms.functional.adjust_brightness(
                img[ind], factor
            )
            img[ind] = np.clip(img[ind], 0, 1)
        return img


class Change_brightness(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = transforms.functional.adjust_brightness(
                img[ind], factor
            )
        return img


class self_normalize(object):
    def __call__(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x
    

class remove_channel(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        img_size = np.array(img).shape
        if min(img_size) < 4:
            return img
        if np.random.rand() <= self.p:
            channel_to_blacken = np.random.choice(
                np.array([0, 2, 3]), 1, replace=False
            )[0]
            img[channel_to_blacken] = torch.zeros(1, *img.shape[1:])
            return img
        else:
            return img
        
        
class NoiseInjection(nn.Module):
    def __init__(self, low=0.785, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(x).uniform_(self.low, self.high)
        return torch.where(x == 1.0, noise, x)