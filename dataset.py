import numpy as np
import os
import pandas as pd
import torch
from glob import glob
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread
from skimage.morphology import disk, binary_erosion, label
from skimage.transform import downscale_local_mean
from torch.utils.data import Dataset

from subsets import data_frame_subset


class TomoDetectionDataset(Dataset):
    """Duke Digital Breast Tomosythesis (DBT) detection dataset"""

    cell_size = 96
    img_width = cell_size * 7
    img_height = cell_size * 11
    out_channels = 5
    grid_size = (img_height // cell_size, img_width // cell_size)
    anchor = (256, 256)

    def __init__(
        self,
        csv_views,
        csv_bboxes,
        root_dir,
        transform=None,
        skip_preprocessing=False,
        downscale=2,
        subset="train",
        random=False,
        only_biopsied=False,
        max_slice_offset=0,
        seed=42
    ):
        """
        :param csv_views: (string) path to csv file with views (see: data/data_train.py)
        :param csv_bboxes: (string) path to csv file with bounding boxes (manual annotations)
        :param root_dir: (string) root folder with PNG images containing folders for patients
        :param transform: transformation to apply to samples (see: transform.py)
        :param skip_preprocessing: set to True if root_dir is set to preprocess.py output folder
        :param downscale: even if skip_proprocessing is set to True, boxes are still downscaled
        :param subset: [test|train|validation]
        :param random: ensure that the same slice is sampled for the same case (useful for validation set)
        :param only_biopsied: filters for cases with boxes
        :param max_slice_offset: range of slices to sample from the central one (0 uses a formula based on box size)
        :param seed: random seed for training-validation set split
        """
        assert subset in ["test", "train", "validation"]
        self.random = random
        self.data_frame = data_frame_subset(
            csv_views, csv_bboxes, subset, seed=seed
        )

        self.df_bboxes = pd.read_csv(csv_bboxes)

        if not only_biopsied:
            self.data_frame = self.data_frame[
                self.data_frame["StudyUID"].isin(self.df_bboxes["StudyUID"])
            ]

        self.root_dir = root_dir
        self.transform = transform
        self.skip_preprocessing = skip_preprocessing
        self.downscale = downscale

        # coordinate conv channels
        self.in_channels = 1

        if max_slice_offset == 0:
            self.df_bboxes["SliceOffset"] = self.df_bboxes.apply(
                lambda row: int(np.sqrt((row["Width"] + row["Height"]) / 2)), axis=1
            )
            if subset == "validation":
                self.df_bboxes["SliceOffset"] = self.df_bboxes["SliceOffset"] // 2
        else:
            self.df_bboxes["SliceOffset"] = int(max_slice_offset)

        self.df_bboxes = self.df_bboxes[
            self.df_bboxes["StudyUID"].isin(set(self.data_frame["StudyUID"]))
        ]

        print(
            "{} boxes for {} studies in {} set".format(
                len(self.df_bboxes), len(set(self.df_bboxes["StudyUID"])), subset
            )
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # read sample data
        pid = str(self.data_frame.iloc[idx]["PatientID"]).zfill(5)
        sid = self.data_frame.iloc[idx]["StudyUID"]
        view = str(self.data_frame.iloc[idx]["View"])

        # filter bboxes related to sample
        df_view_bboxes = self.df_bboxes[
            (self.df_bboxes["StudyUID"] == sid)
            & (self.df_bboxes["View"] == view.lower())
        ]

        # find the number of slices
        max_slice = self._max_slice(pid, sid, view)

        if not self.random:
            # assure the same slice for samples if random is set to False
            np.random.seed(idx)

        slice_n = np.random.randint(max_slice + 1)

        # sample slice for positive case
        if len(df_view_bboxes) > 0:
            box = df_view_bboxes.sample()
            slice_n = box.iloc[0]["Slice"]  # GT central slice
            max_slice_offset = box.iloc[0]["SliceOffset"]
            offset = np.random.randint(-max_slice_offset, max_slice_offset + 1)
            slice_n = slice_n + offset
            slice_n = max(0, min(max_slice, slice_n))
            # we take all boxes from slices "close" to the sampled one
            df_view_bboxes = df_view_bboxes[
                abs(slice_n - df_view_bboxes["Slice"]) <= df_view_bboxes["SliceOffset"]
            ]

        # read image
        image_name = "{}TomosynthesisReconstruction_{}_.png".format(
            view.upper(), slice_n
        )
        image_path = os.path.join(self.root_dir, pid, sid, image_name)

        if self.skip_preprocessing:
            img = imread(image_path)
        else:
            img = self._imread(image_path, flip="R" in view.upper())

        # read boxes
        boxes = self._df2dict(df_view_bboxes)

        if self.transform is not None:
            img, boxes = self.transform((img, boxes))

        lbl = self._boxes2label(boxes)

        # normalize
        img = img.astype(np.float32) / np.max(img)
        # fix dimensions (N, C, H, W)
        img = img[..., np.newaxis]
        img = img.transpose((2, 0, 1))

        # cast to tensors
        img_tensor = torch.from_numpy(img)
        lbl_tensor = torch.from_numpy(lbl)

        return img_tensor, lbl_tensor

    def _max_slice(self, pid, sid, view):
        view_template = "{}TomosynthesisReconstruction_*_.png".format(view.upper())
        view_files = glob(os.path.join(self.root_dir, pid, sid, view_template))
        max_slice = np.max([int(x.split("_")[-2]) for x in view_files])
        return max_slice

    def _imread(self, imgpath, flip=False):
        image = imread(imgpath)
        if self.downscale != 1:
            image = downscale_local_mean(image, (self.downscale, self.downscale))
        if flip:
            image = np.fliplr(image).copy()
        image = self._preprocess(image)
        return image

    def _preprocess(self, image, erosion=5):
        mask = self._mask(image, erosion=erosion)
        image = mask * image
        return image

    def _mask(self, image, erosion=10):
        mask = image > 0
        mask = np.pad(mask, ((0, 0), (1, 0)), mode="constant", constant_values=1)
        mask = binary_fill_holes(mask)
        mask = mask[:, 1:]
        mask = binary_erosion(mask, disk(erosion))
        cc = label(mask, background=0)
        lcc = np.argmax(np.bincount(cc.flat)[1:]) + 1
        mask = cc == lcc
        return mask

    def _mean_filter(self, image, filter_size=4):
        fs = filter_size
        yy, xx = np.nonzero(image >= np.max(image) * 0.99)
        image_out = image
        for y, x in zip(yy, xx):
            neighborhood = image[max(0, y - fs) : y + fs, max(0, x - fs) : x + fs]
            image_out[y, x] = np.mean(neighborhood)
        return image_out

    def _df2dict(self, df_view_boxes):
        df_boxes = df_view_boxes.copy()
        df_boxes = df_boxes[["X", "Y", "Width", "Height"]]
        df_boxes["Width"] = df_boxes["Width"] // self.downscale
        df_boxes["Height"] = df_boxes["Height"] // self.downscale
        df_boxes["X"] = df_boxes["X"] // self.downscale
        df_boxes["Y"] = df_boxes["Y"] // self.downscale
        df_boxes["X"] = df_boxes["X"] + (df_boxes["Width"] // 2)
        df_boxes["Y"] = df_boxes["Y"] + (df_boxes["Height"] // 2)
        return df_boxes.to_dict(orient="list")

    def _boxes2label(self, boxes):
        label = np.zeros((self.out_channels,) + self.grid_size, dtype=np.float32)
        csz = self.cell_size
        box_indices = range(len(boxes["X"]))
        if "Points" in boxes:
            box_indices = zip(box_indices, boxes["Points"])
            box_indices = sorted(box_indices, key=lambda i: i[1])
            box_indices = [i[0] for i in box_indices]
        for b in box_indices:
            # box dimensions
            w = boxes["Width"][b]
            h = boxes["Height"][b]
            # box center point
            x = boxes["X"][b]
            y = boxes["Y"][b]
            # fill label tensor
            pos_cell_x = min(self.grid_size[1] - 1, int(x / csz))
            pos_cell_y = min(self.grid_size[0] - 1, int(y / csz))
            label[0, pos_cell_y, pos_cell_x] = 1.0
            y_offset = ((y % csz) - (csz / 2)) / (csz / 2)
            x_offset = ((x % csz) - (csz / 2)) / (csz / 2)
            label[1, pos_cell_y, pos_cell_x] = y_offset
            label[2, pos_cell_y, pos_cell_x] = x_offset
            y_scale = np.sqrt(float(h) / self.anchor[0])
            x_scale = np.sqrt(float(w) / self.anchor[1])
            label[3, pos_cell_y, pos_cell_x] = y_scale
            label[4, pos_cell_y, pos_cell_x] = x_scale
        return label
