import argparse
import os
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mlflow import log_metric, log_param, get_artifact_uri
from skimage.io import imsave
from sklearn.model_selection import ParameterGrid
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TomoDetectionDataset as Dataset
from dense_yolo import DenseYOLO
from loss import objectness_module, LocalizationLoss
from sampler import TomoBatchSampler
from transform import transforms


def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    hparams_dict = {
        "block_config": [(1, 3, 2, 6, 4), (2, 6, 4, 12, 8)],
        "num_init_features": [8, 16],
        "growth_rate": [8, 16],
        "bn_size": [2, 4],
    }
    hparams = list(ParameterGrid(hparams_dict))  # 16 configs

    loss_params_dict = [
        {"loss": ["CE", "weighted-CE"], "alpha": [0.25, 0.5, 1.0]},  # 6 configs
        {"loss": ["focal"], "alpha": [0.25, 0.5, 1.0], "gamma": [0.5, 1.0, 2.0]},  # 9 configs
        {
            "loss": ["reduced-focal"],
            "alpha": [0.25, 0.5, 1.0],
            "gamma": [0.5, 1.0, 2.0],
            "reduce_th": [0.5],
        }  # 9 configs
    ]  # 24 configs
    loss_params = list(ParameterGrid(loss_params_dict))

    loss_params = loss_params * 2  # 48 configs

    try:
        mlflow.set_tracking_uri(args.mlruns_path)
        experiment_id = (
            args.experiment_id
            if args.experiment_id
            else mlflow.create_experiment(name=args.experiment_name)
        )
    except Exception as _:
        print("experiment-id must be unique")
        return

    for i, loss_param in tqdm(enumerate(loss_params)):

        for j, hparam in enumerate(hparams):

            with mlflow.start_run(experiment_id=experiment_id):
                mlflow_log_params(loss_param, hparam)

                try:
                    yolo = DenseYOLO(img_channels=1, out_channels=Dataset.out_channels, **hparam)
                    yolo.to(device)

                    objectness_loss = objectness_module(
                        name=loss_param["loss"], args=argparse.Namespace(**loss_param)
                    )
                    localization_loss = LocalizationLoss(weight=args.loc_weight)

                    optimizer = optim.Adam(yolo.parameters(), lr=args.lr)

                    early_stop = args.patience
                    run_tpr2 = 0.0
                    run_tpr1 = 0.0
                    run_auc = 0.0

                    for _ in range(args.epochs):

                        if early_stop == 0:
                            break

                        for phase in ["train", "valid"]:
                            if phase == "train":
                                yolo.train()
                                early_stop -= 1
                            else:
                                yolo.eval()

                            df_validation_pred = pd.DataFrame()
                            valid_target_nb = 0

                            for data in loaders[phase]:
                                x, y_true = data
                                x, y_true = x.to(device), y_true.to(device)

                                optimizer.zero_grad()

                                with torch.set_grad_enabled(phase == "train"):
                                    y_pred = yolo(x)

                                    obj = objectness_loss(y_pred, y_true)
                                    loc = localization_loss(y_pred, y_true)
                                    total_loss = obj + loc

                                    if phase == "train":
                                        total_loss.backward()
                                        clip_grad_norm_(yolo.parameters(), 0.5)
                                        optimizer.step()
                                    else:
                                        y_true_np = y_true.detach().cpu().numpy()
                                        valid_target_nb += np.sum(y_true_np[:, 0])
                                        df_batch_pred = evaluate_batch(y_pred, y_true)
                                        df_validation_pred = df_validation_pred.append(
                                            df_batch_pred, ignore_index=True, sort=False
                                        )

                            if phase == "valid":
                                tpr, fps = froc(df_validation_pred, valid_target_nb)
                                epoch_tpr2 = np.interp(2.0, fps, tpr)
                                epoch_tpr1 = np.interp(1.0, fps, tpr)
                                if epoch_tpr2 > run_tpr2:
                                    early_stop = args.patience
                                    run_tpr2 = epoch_tpr2
                                    run_tpr1 = epoch_tpr1
                                    run_auc = np.trapz(tpr, fps)
                                    torch.save(
                                        yolo.state_dict(),
                                        os.path.join(get_artifact_uri(), "yolo.pt"),
                                    )
                                    imsave(
                                        os.path.join(get_artifact_uri(), "froc.png"),
                                        plot_froc(fps, tpr),
                                    )

                    log_metric("TPR2", run_tpr2)
                    log_metric("TPR1", run_tpr1)
                    log_metric("AUC", run_auc)

                except Exception as e:
                    print(
                        "{:0>2d}/{} | {} {}".format(
                            j + 1, len(hparams), hparams[j], type(e).__name__
                        )
                    )


def mlflow_log_params(loss_param, hparam):
    for key in loss_param:
        log_param(key, loss_param[key])
    log_param("loss_fun", str(loss_param))
    for key in hparam:
        log_param(key, hparam[key])
    log_param("network", str(hparam))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)
    sampler_train = TomoBatchSampler(
        batch_size=args.batch_size, data_frame=dataset_train.data_frame
    )

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset="train",
        random=True,
        only_biopsied=args.only_biopsied,
        transform=transforms(train=True),
        skip_preprocessing=True,
        downscale=args.downscale,
        max_slice_offset=args.slice_offset,
        seed=args.seed,
    )
    valid = Dataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset="validation",
        random=False,
        transform=transforms(train=False),
        skip_preprocessing=True,
        downscale=args.downscale,
        max_slice_offset=args.slice_offset,
        seed=args.seed,
    )
    return train, valid


def froc(df, targets_nb):
    total_slices = len(df.drop_duplicates(subset=["PID"]))
    total_tps = targets_nb
    tpr = [0.0]
    fps = [0.0]
    max_fps = 4.0
    thresholds = sorted(df[df["TP"] == 1]["Score"], reverse=True)
    for th in thresholds:
        df_th = df[df["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(subset=["PID", "TP", "GTID"])
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_tps
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_slices
        if fps_th > max_fps:
            tpr.append(tpr[-1])
            fps.append(max_fps)
            break
        tpr.append(tpr_th)
        fps.append(fps_th)
    if np.max(fps) < max_fps:
        tpr.append(tpr[-1])
        fps.append(max_fps)
    return tpr, fps


def plot_froc(fps, tpr, color="darkorange", linestyle="-"):
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)
    plt.plot(fps, tpr, color=color, linestyle=linestyle, lw=2)
    plt.xlim([0.0, 4.0])
    plt.xticks(np.arange(0.0, 4.5, 0.5))
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.xlabel("Mean FPs per slice", fontsize=24)
    plt.ylabel("Sensitivity", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def is_tp(pred_box, true_box, min_dist=50):
    # box: center point + dimensions
    pred_y, pred_x = pred_box["Y"], pred_box["X"]
    gt_y, gt_x = true_box["Y"], true_box["X"]
    # distance between GT and predicted center points
    dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    # TP radius based on GT box size
    dist_threshold = np.sqrt(true_box["Width"] ** 2 + true_box["Height"] ** 2) / 2.
    dist_threshold = max(dist_threshold, min_dist)
    # TP if predicted center within GT radius
    return dist <= dist_threshold


def evaluate_batch(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    df_eval = pd.DataFrame()
    for i in range(y_pred.shape[0]):
        df_gt_boxes = pred2boxes(y_true[i], threshold=1.0)
        df_gt_boxes["GTID"] = np.random.randint(10e10) * (1 + df_gt_boxes["X"])
        df_pred_boxes = pred2boxes(y_pred[i])
        df_pred_boxes["PID"] = np.random.randint(10e12)
        df_pred_boxes["TP"] = 0
        df_pred_boxes["GTID"] = np.random.choice(
            list(set(df_gt_boxes["GTID"])), df_pred_boxes.shape[0]
        )
        for index, pred_box in df_pred_boxes.iterrows():
            tp_list = [
                (j, is_tp(pred_box, x_box)) for j, x_box in df_gt_boxes.iterrows()
            ]
            if any([tp[1] for tp in tp_list]):
                tp_index = [tp[0] for tp in tp_list if tp[1]][0]
                df_pred_boxes.at[index, "TP"] = 1
                df_pred_boxes.at[index, "GTID"] = df_gt_boxes.at[tp_index, "GTID"]
        df_eval = df_eval.append(df_pred_boxes, ignore_index=True, sort=False)
    return df_eval


def pred2boxes(pred, threshold=None):
    # box: center point + dimensions
    anchor = Dataset.anchor
    cell_size = Dataset.cell_size
    np.nan_to_num(pred, copy=False)
    obj_th = pred[0]
    if threshold is None:
        threshold = min(0.001, np.max(obj_th) * 0.5)
    obj_th[obj_th < threshold] = 0
    yy, xx = np.nonzero(obj_th)
    scores = []
    xs = []
    ys = []
    ws = []
    hs = []
    for i in range(len(yy)):
        scores.append(pred[0, yy[i], xx[i]])
        h = int(anchor[0] * pred[3, yy[i], xx[i]] ** 2)
        hs.append(h)
        w = int(anchor[1] * pred[4, yy[i], xx[i]] ** 2)
        ws.append(w)
        y_offset = pred[1, yy[i], xx[i]]
        y_mid = yy[i] * cell_size + (cell_size / 2) + (cell_size / 2) * y_offset
        ys.append(int(y_mid))
        x_offset = pred[2, yy[i], xx[i]]
        x_mid = xx[i] * cell_size + (cell_size / 2) + (cell_size / 2) * x_offset
        xs.append(int(x_mid))

    df_dict = {"Score": scores, "X": xs, "Y": ys, "Width": ws, "Height": hs}
    df_boxes = pd.DataFrame(df_dict)
    df_boxes.sort_values(by="Score", ascending=False, inplace=True)
    return df_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyper-parameters grid search for YOLO model for cancer detection in Duke DBT volumes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="early stopping: number of epochs to wait for improvement (default: 25)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--loc-weight",
        type=float,
        default=0.5,
        help="weight of localization loss (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="device for training (default: cuda:1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--data-views",
        type=str,
        default="/data/data_train_v2.csv",
        help="csv file listing training views together with category label",
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
        default="/data/TomoImagesPP/",
        help="root folder with preprocessed images",
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
        help="input image downscale factor (default 2)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="0",
        help="experiment name for new mlflow (default: 0)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="experiment id to restore in-progress mlflow experiment (default: None)",
    )
    parser.add_argument(
        "--mlruns-path",
        type=str,
        default="/data/mlruns",
        help="path for mlflow results (default: /data/mlruns)",
    )
    parser.add_argument(
        "--slice-offset",
        type=int,
        default=0,
        help="maximum offset from central slice to consider as GT bounding box (default: 0)",
    )
    parser.add_argument(
        "--only-biopsied",
        default=True,  # set to true by default for convenience
        action="store_true",
        help="flag to use only biopsied cases",
    )
    args = parser.parse_args()
    main(args)
