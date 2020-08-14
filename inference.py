import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
from glob import glob
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread
from skimage.morphology import disk, binary_erosion, label
from skimage.transform import downscale_local_mean
from tqdm import tqdm

from dataset import TomoDetectionDataset as Dataset
from dense_yolo import DenseYOLO
from subsets import data_frame_subset

cell_size = Dataset.cell_size
# larger grid size for inference to run inference on full image without cropping
img_height = cell_size * 12
img_width = cell_size * 9
grid_size = (img_height // cell_size, img_width // cell_size)
anchor = Dataset.anchor


def main(args, config):
    data_frame = data_frame_subset(
        args.data_views, args.data_boxes, args.subset, seed=args.seed
    )
    pred_data_frame = pd.DataFrame()

    if args.only_biopsied:
        data_frame = data_frame[(data_frame["Benign"] == 1) | (data_frame["Cancer"] == 1)]

    with torch.set_grad_enabled(False):
        yolo = DenseYOLO(img_channels=1, out_channels=Dataset.out_channels, **config)

        if args.multi_gpu and torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
            yolo = torch.nn.DataParallel(yolo)
        else:
            device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

        yolo.to(device)

        state_dict = torch.load(args.weights)
        yolo.load_state_dict(state_dict)
        yolo.eval()
        yolo.to(device)

        for index, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
            pid = str(row["PatientID"]).zfill(5)
            sid = row["StudyUID"]
            view = str(row["View"])

            view_template = "{}TomosynthesisReconstruction_*_.png".format(view.upper())
            view_files = glob(os.path.join(args.images, pid, sid, view_template))

            batch = []
            volume = []
            pred_view = np.zeros((len(view_files), 5) + grid_size)

            for slice_n in range(len(view_files)):
                batch.append(
                    read_slice_image(
                        pid, sid, view, slice_n, args.images, args.downscale
                    )
                )
                volume.append(batch[-1][0])

                if len(batch) >= args.batch_size:
                    y_pred = predict(yolo, batch, device)
                    pred_view[slice_n + 1 - len(batch) : slice_n + 1] = y_pred
                    batch = []

            if len(batch) > 0:
                y_pred = predict(yolo, batch, device)
                pred_view[-len(batch) :] = y_pred

            pred_view = average_predictions(pred_view, view_split=args.view_split)
            if args.keep_splits > 0:
                pred_view = filter_by_score(pred_view, keep=args.keep_splits)

            slice_span = len(volume) / args.view_split
            df_view_bboxes = pred2bboxes(
                pred_view, slice_span=slice_span, threshold=args.pred_threshold
            )

            df_view_bboxes = remove_empty_boxes(df_view_bboxes, np.array(volume))

            df_view_bboxes["PatientID"] = pid
            df_view_bboxes["StudyUID"] = sid
            df_view_bboxes["View"] = view
            pred_data_frame = pred_data_frame.append(
                df_view_bboxes, ignore_index=True, sort=False
            )

    # rescale boxes to original images size
    pred_data_frame["X"] = pred_data_frame["X"] * args.downscale
    pred_data_frame["Y"] = pred_data_frame["Y"] * args.downscale
    pred_data_frame["Width"] = pred_data_frame["Width"] * args.downscale
    pred_data_frame["Height"] = pred_data_frame["Height"] * args.downscale

    pred_data_frame = pred_data_frame[
        ["PatientID", "StudyUID", "View", "Score", "Z", "X", "Y", "Depth", "Width", "Height"]
    ]
    pred_data_frame[["X", "Y", "Z", "Width", "Height", "Depth"]] = pred_data_frame[
        ["X", "Y", "Z", "Width", "Height", "Depth"]
    ].astype(int)

    pred_data_frame.to_csv(args.predictions, index=False)


def predict(model, batch, device):
    batch_tensor = torch.from_numpy(np.array(batch))
    batch_tensor = batch_tensor.to(device)
    y_pred_device = model(batch_tensor)
    y_pred = y_pred_device.cpu().numpy()
    return np.squeeze(y_pred)


def read_slice_image(pid, sid, view, slice_n, images_dir, downscale):
    filename = "{}TomosynthesisReconstruction_{}_.png".format(view.upper(), slice_n)
    image_path = os.path.join(images_dir, pid, sid, filename)
    img = _imread(image_path, downscale=downscale, flip="R" in view.upper())

    if img.shape[0] < img_height:
        pad_y = img_height - img.shape[0]
        img = np.pad(img, ((0, pad_y), (0, 0)), mode="constant")
    elif img.shape[0] > img_height:
        img = img[:img_height, :]

    if img.shape[1] < img_width:
        pad_x = img_width - img.shape[1]
        img = np.pad(img, ((0, 0), (0, pad_x)), mode="constant")
    elif img.shape[1] > img_width:
        img = img[:, :img_width]

    # normalize
    img = img.astype(np.float32) / np.max(img)

    # fix dimensions (N, C, H, W)
    img = img[..., np.newaxis]
    img = img.transpose((2, 0, 1))

    return img


def _imread(imgpath, downscale, flip=False):
    image = imread(imgpath)
    if downscale != 1:
        image = downscale_local_mean(image, (downscale, downscale))
    if flip:
        image = np.fliplr(image).copy()
    image = _preprocess(image)
    return image


def _preprocess(image, erosion=5):
    mask = _mask(image, erosion=erosion)
    image = mask * image
    return image


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


def average_predictions(pred_view, view_split=4):
    pred_view_avg = np.zeros((view_split, 5) + grid_size)
    slice_span = int(pred_view.shape[0] / view_split)
    for i in range(view_split):
        pred_view_avg[i] = np.mean(
            pred_view[i * slice_span : (i + 1) * slice_span], axis=0
        )
    return pred_view_avg


def filter_by_score(pred_view, keep):
    if keep >= pred_view.shape[0]:
        return pred_view
    for i in range(pred_view.shape[-2]):
        for j in range(pred_view.shape[-1]):
            pred_cell = pred_view[:, 0, i, j]
            threshold = sorted(pred_cell.flat, reverse=True)[keep]
            for k in range(pred_view.shape[0]):
                if pred_view[k, 0, i, j] <= threshold:
                    pred_view[k, 0, i, j] = 0.0
    return pred_view


def pred2bboxes(pred, slice_span, threshold=None):
    # box: upper-left corner + width + height + first slice + depth
    np.nan_to_num(pred, copy=False)
    obj_th = pred[:, 0, ...]
    if threshold is None:
        threshold = min(0.0001, np.max(obj_th) * 0.5)
    obj_th[obj_th < threshold] = 0
    z, y, x = np.nonzero(obj_th)
    scores = []
    xs = []
    ys = []
    hs = []
    ws = []
    for i in range(len(z)):
        scores.append(pred[z[i], 0, y[i], x[i]])
        h = int(anchor[0] * pred[z[i], 3, y[i], x[i]] ** 2)
        hs.append(h)
        w = int(anchor[0] * pred[z[i], 4, y[i], x[i]] ** 2)
        ws.append(w)
        y_offset = pred[z[i], 1, y[i], x[i]]
        y_mid = y[i] * cell_size + (cell_size / 2) + (cell_size / 2) * y_offset
        ys.append(int(y_mid - h / 2))
        x_offset = pred[z[i], 2, y[i], x[i]]
        x_mid = x[i] * cell_size + (cell_size / 2) + (cell_size / 2) * x_offset
        xs.append(int(x_mid - w / 2))

    zs = [s * slice_span for s in z]
    df_dict = {
        "Z": zs,
        "X": xs,
        "Y": ys,
        "Width": ws,
        "Height": hs,
        "Depth": [slice_span] * len(zs),
        "Score": scores,
    }
    df_bboxes = pd.DataFrame(df_dict)
    df_bboxes.sort_values(by="Score", ascending=False, inplace=True)
    return df_bboxes


def remove_empty_boxes(df, volume):
    # box: upper-left corner + width + height + first slice + depth
    empty_indices = []
    for index, box in df.iterrows():
        w = int(box["Width"])
        h = int(box["Height"])
        d = int(box["Depth"])
        x = int(max(box["X"], 0))
        y = int(max(box["Y"], 0))
        z = int(max(box["Z"], 0))
        box_volume = volume[z : z + d, y : y + h, x : x + w]
        if np.sum(box_volume == 0) > 0.5 * w * h * d:
            empty_indices.append(index)
    df = df.drop(index=empty_indices)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running inference using trained YOLO model for cancer detection in Duke DBT volumes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for testing (default: cuda:1)",
    )
    parser.add_argument(
        "--multi-gpu",
        default=False,
        action="store_true",
        help="flag to train on multiple gpus",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for testing (default: 16)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_default.json",
        help="network config file (see: config_default.json)",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="file with saved weights"
    )
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
        "--images",
        type=str,
        default="/data/TomoImages/",
        help="root folder with images",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="output file path with predictions",
    )
    parser.add_argument(
        "--view-split",
        type=int,
        default=2,
        help="number of view parts for averaging predictions (default: 2)",
    )
    parser.add_argument(
        "--keep-splits",
        type=int,
        default=0,
        help="number of averaged view splits to keep after filtering (default: 0=view-split)",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.0001,
        help="threshold for minimum box prediction confidence (default: 0.0001)",
    )
    parser.add_argument(
        "--only-biopsied",
        default=False,
        action="store_true",
        help="flag to use only biopsied cases",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="validation",
        help="subset to run inference on [all|train|validation|test] (default: validation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for validation split (default: 42)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="input image downscale factor used to upscale boxes to original scale (default 2)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = json.load(fp)

    main(args, config)
