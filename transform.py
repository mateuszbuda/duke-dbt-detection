import numpy as np
from skimage.transform import rescale
from torchvision.transforms import Compose

from dataset import TomoDetectionDataset


def transforms(train=True):
    if train:
        return Compose(
            [Crop((TomoDetectionDataset.img_height, TomoDetectionDataset.img_width))]
        )
    else:
        return Crop(
            (TomoDetectionDataset.img_height, TomoDetectionDataset.img_width),
            random=False,
        )


class Scale(object):

    def __init__(self, scale):
        assert isinstance(scale, (float, tuple))
        if isinstance(scale, float):
            assert 0.0 < scale < 1.0
            self.scale = (1.0 - scale, 1.0 + scale)
        else:
            assert len(scale) == 2
            assert 0.0 < scale[0] < scale[1]
            self.scale = scale

    def __call__(self, sample):
        image, boxes = sample

        # don't augment normal cases
        if len(boxes["X"]) == 0:
            return image, boxes

        sample_scale = np.random.rand()
        sample_scale = sample_scale * (self.scale[1] - self.scale[0]) + self.scale[0]

        scaled = rescale(
            image, sample_scale, multichannel=True, mode="constant", anti_aliasing=False
        )

        boxes["X"] = [int(x * sample_scale) for x in boxes["X"]]
        boxes["Y"] = [int(y * sample_scale) for y in boxes["Y"]]
        boxes["Width"] = [int(w * sample_scale) for w in boxes["Width"]]
        boxes["Height"] = [int(h * sample_scale) for h in boxes["Height"]]

        return scaled, boxes


class Crop(object):

    def __init__(self, crop_size, random=True):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.random = random

    def __call__(self, sample):
        image, boxes = sample

        h = image.shape[0]
        w = image.shape[1]
        y_max = max(h - self.crop_size[0], 1)
        x_max = max(w - self.crop_size[1], 1) // 2
        if image[h // 2, self.crop_size[1]] == 0:
            x_max //= 2
        y_min = x_min = 0
        x_max_box = 0

        # don't crop boxes
        margin = 16
        if len(boxes["X"]) > 0:
            y_min_box = np.min(np.array(boxes["Y"]) - np.array(boxes["Height"]) // 2)
            x_min_box = np.min(np.array(boxes["X"]) - np.array(boxes["Width"]) // 2)
            y_max_box = np.max(np.array(boxes["Y"]) + np.array(boxes["Height"]) // 2)
            x_max_box = np.max(np.array(boxes["X"]) + np.array(boxes["Width"]) // 2)
            y_min = max(y_min, min(h, y_max_box + margin) - self.crop_size[0])
            x_min = max(x_min, min(w, x_max_box + margin) - self.crop_size[1])
            y_max = min(y_max, max(0, y_min_box - margin))
            x_max = min(x_max, max(0, x_min_box - margin))
            if x_max <= x_min:
                x_max = x_min + 1
            if y_max <= y_min:
                y_max = y_min + 1

        if self.random:
            y_offset = np.random.randint(y_min, y_max)
            x_offset = np.random.randint(x_min, x_max)
        else:
            y_offset = (y_min + y_max) // 2
            if x_max_box + margin < self.crop_size[1]:
                x_offset = 0
            else:
                x_offset = (x_min + x_max) // 2

        cropped = image[
            y_offset : y_offset + self.crop_size[0],
            x_offset : x_offset + self.crop_size[1],
        ]

        # don't let empty crop
        if np.max(cropped) == 0:
            y_offset = y_max // 2
            x_offset = 0
            cropped = image[
                y_offset : y_offset + self.crop_size[0],
                x_offset : x_offset + self.crop_size[1],
            ]

        boxes["X"] = [max(0, x - x_offset) for x in boxes["X"]]
        boxes["Y"] = [max(0, y - y_offset) for y in boxes["Y"]]

        return cropped, boxes
