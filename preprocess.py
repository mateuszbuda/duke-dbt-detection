import argparse
import numpy as np
import os
from glob import glob
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread, imsave
from skimage.morphology import disk, binary_erosion, label
from skimage.transform import downscale_local_mean
from tqdm import tqdm

from dataset import TomoDetectionDataset


def main(args):
    dataset = TomoDetectionDataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset=args.subset,
        only_biopsied=args.only_biopsied,
    )
    data_frame = dataset.data_frame

    for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        pid = str(row["PatientID"]).zfill(5)
        sid = row["StudyUID"]
        view = str(row["View"])

        view_template = "{}TomosynthesisReconstruction_*_.png".format(view.upper())
        view_files = glob(os.path.join(args.images, pid, sid, view_template))

        dst_dir = os.path.join(args.output, pid, sid)
        os.makedirs(dst_dir, exist_ok=True)

        for slice_n in range(len(view_files)):
            slice_image, filename = read_slice_image(
                pid, sid, view, slice_n, args.images, downscale=args.downscale
            )
            slice_image = _preprocess(slice_image)
            imsave(os.path.join(dst_dir, filename), slice_image)


def read_slice_image(pid, sid, view, slice_n, images_dir, downscale=2):
    filename = "{}TomosynthesisReconstruction_{}_.png".format(view.upper(), slice_n)
    image_path = os.path.join(images_dir, pid, sid, filename)
    img = _imread(image_path, flip="R" in view.upper(), downscale=downscale)
    return img, filename


def _imread(imgpath, flip=False, downscale=2):
    image = imread(imgpath)
    if downscale != 1:
        image = downscale_local_mean(image, (downscale, downscale))
    if flip:
        image = np.fliplr(image).copy()
    image = _preprocess(image)
    return image


def _preprocess(image, erosion=5):
    mask = _mask(image, erosion=erosion)
    image = image * mask
    return image.astype(np.uint16)


def _mask(image, erosion=10):
    mask = image > 0
    mask = np.pad(mask, ((0, 0), (1, 0)), mode="constant", constant_values=1)
    mask = binary_fill_holes(mask)
    mask = mask[:, 1:]
    mask = binary_erosion(mask, disk(erosion))
    cc = label(mask, background=0)
    lcc = np.argmax(np.bincount(cc.flat)[1:]) + 1
    mask = cc == lcc
    return mask


def _mean_filter(image, filter_size=4):
    fs = filter_size
    yy, xx = np.nonzero(image >= np.max(image) * 0.99)
    image_out = image
    for y, x in zip(yy, xx):
        neighborhood = image[max(0, y - fs) : y + fs, max(0, x - fs) : x + fs]
        image_out[y, x] = np.mean(neighborhood)
    return image_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-processing of DBT images")
    parser.add_argument(
        "--data-views",
        type=str,
        default="/data/data_train_v2.csv",
        help="csv file listing training/test views together with category label",
    )
    parser.add_argument(
        "--data-boxes",
        type=str,
        default="/data/bboxes_v2.csv",
        help="csv file defining ground truth bounding boxes",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset to run preprocessing on [all|train|validation|test] (default: all)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="/data/TomoImages/",
        help="root folder with preprocessed images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/TomoImagesPP/",
        help="Output folder for saving pre-processed images",
    )
    parser.add_argument(
        "--only-biopsied",
        default=False,
        action="store_true",
        help="flag to run preprocessing only on biopsied cases",
    )
    parser.add_argument(
        "--downscale", type=int, default=2, help="Downscale factor (default: 2)"
    )
    args = parser.parse_args()
    main(args)
