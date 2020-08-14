import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def main(args):
    df_gt = pd.read_csv(args.gt)
    df_pred = pd.read_csv(args.predictions)

    print("Num cases in predictions: " + str(len(set(df_pred["PatientID"]))))
    print("Num studies in predictions: " + str(len(set(df_pred["StudyUID"]))))
    total_volumes = len(df_pred.drop_duplicates(subset=["StudyUID", "View"]))
    print("Num volumes in predictions: " + str(total_volumes))

    df_gt = df_gt[df_gt["Spot"] == 0]
    df_gt = df_gt[df_gt["StudyUID"].isin(set(df_pred["StudyUID"]))]

    print("Num cases in GT: " + str(len(set(df_gt["PatientID"]))))
    print("Num studies in GT: " + str(len(set(df_gt["StudyUID"]))))
    print("Num cancer boxes in GT: " + str(len(df_gt[df_gt["Class"] == "cancer"])))
    print("Num benign boxes in GT: " + str(len(df_gt[df_gt["Class"] == "benign"])))
    print("Num all boxes in GT: " + str(len(df_gt)))

    df_pred["TP"] = 0
    df_pred["GTID"] = 0
    positive_ids = set(df_gt["PatientID"])
    thresholds = [1.]

    for index, pred_box in tqdm(df_pred.iterrows(), total=len(df_pred)):
        if pred_box["PatientID"] not in positive_ids:
            continue
        df_gt_view = df_gt[
            (df_gt["StudyUID"] == pred_box["StudyUID"])
            & (df_gt["View"] == pred_box["View"])
        ]
        df_pred_view = df_pred[
            (df_pred["StudyUID"] == pred_box["StudyUID"])
            & (df_pred["View"] == pred_box["View"])
        ]
        gt_slice_offset = np.max(df_pred_view["Z"] + df_pred_view["Depth"])
        if not args.two_dim:
            gt_slice_offset = gt_slice_offset / 4.0
        is_tp_list = [
            (i, is_tp(x_box, pred_box, slice_offset=gt_slice_offset))
            for i, x_box in df_gt_view.iterrows()
        ]
        tps = [x for x in is_tp_list if x[1]]
        if len(tps) > 0:
            df_pred.at[index, "TP"] = 1
            df_pred.at[index, "GTID"] = tps[0][0]
            thresholds.append(pred_box["Score"])

    thresholds.append(0.)

    df_pred.to_csv(args.predictions.replace(".csv", "_eval.csv"), index=False)

    tpr, fps = froc_curve(
        df_gt.copy(),
        df_pred.copy(),
        thresholds,
        verbose=args.verbose,
        cases="all",
        per_side=args.per_side,
    )
    print("All:")
    print("Sensitivity at 1 FP/V = {}".format(sensitivity(fps, tpr, at=1.0)))
    print("Sensitivity at 2 FP/V = {}".format(sensitivity(fps, tpr, at=2.0)))
    print("Sensitivity at 4 FP/V = {}".format(sensitivity(fps, tpr, at=4.0)))
    print("Sensitivity at 6 FP/V = {}".format(sensitivity(fps, tpr, at=6.0)))
    plt.figure(figsize=(20, 15))
    plot_froc(fps, tpr, color="skyblue")

    if "cancer" in (set(df_gt["Class"])):
        tpr, fps = froc_curve(
            df_gt.copy(),
            df_pred.copy(),
            thresholds,
            cases="cancer",
            verbose=args.verbose,
            per_side=args.per_side,
        )
        print("Cancer:")
        print("Sensitivity at 1 FP/V = {}".format(sensitivity(fps, tpr, at=1.0)))
        print("Sensitivity at 2 FP/V = {}".format(sensitivity(fps, tpr, at=2.0)))
        print("Sensitivity at 4 FP/V = {}".format(sensitivity(fps, tpr, at=4.0)))
        print("Sensitivity at 6 FP/V = {}".format(sensitivity(fps, tpr, at=6.0)))
        plot_froc(fps, tpr, color="tomato", linestyle="--")

    if "benign" in (set(df_gt["Class"])):
        tpr, fps = froc_curve(
            df_gt.copy(),
            df_pred.copy(),
            thresholds,
            cases="benign",
            verbose=args.verbose,
            per_side=args.per_side,
        )
        print("Benign:")
        print("Sensitivity at 1 FP/V = {}".format(sensitivity(fps, tpr, at=1.0)))
        print("Sensitivity at 2 FP/V = {}".format(sensitivity(fps, tpr, at=2.0)))
        print("Sensitivity at 4 FP/V = {}".format(sensitivity(fps, tpr, at=4.0)))
        print("Sensitivity at 6 FP/V = {}".format(sensitivity(fps, tpr, at=6.0)))
        plot_froc(fps, tpr, color="limegreen", linestyle="--")

    plot_grid()
    plt.savefig(args.figure, transparent=True)

    print("End " + args.predictions)
    print("")


def froc_auc(fps, tpr, high=4.0):
    x = [0.0]
    y = [0.0]
    for i in range(len(fps)):
        if fps[i] < high:
            x.append(fps[i])
            y.append(tpr[i])
    x.append(high)
    y.append(y[-1])
    return np.trapz(y, x)


def sensitivity(fps, tpr, at=2.0):
    return np.interp(at, fps, tpr)


def plot_froc(fps, tpr, max_fp=8.0, color="darkorange", linestyle="-"):
    plt.plot(fps, tpr, color=color, linestyle=linestyle, lw=4)
    plt.xlim([0.0, max_fp])
    plt.xticks(np.arange(0.0, max_fp + 1.0, 1.0))
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.tick_params(axis="both", which="major", labelsize=32)
    plt.xlabel("Mean FPs per DBT volume", fontsize=42)
    plt.ylabel("Sensitivity", fontsize=42)


def plot_grid():
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=2)
    plt.tight_layout()


def is_tp(gt, pred, slice_offset=10, min_dist=100):
    # box: upper-left corner + width + height
    # 3D: first slice + depth
    pred_y = pred["Y"] + pred["Height"] / 2
    pred_x = pred["X"] + pred["Width"] / 2
    pred_z = pred["Z"] + pred["Depth"] / 2
    gt_y = gt["Y"] + gt["Height"] / 2
    gt_x = gt["X"] + gt["Width"] / 2
    gt_z = gt["Slice"]
    # distance between GT and predicted center points
    dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    # TP radius based on GT box size
    dist_threshold = np.sqrt(gt["Width"] ** 2 + gt["Height"] ** 2) / 2.
    dist_threshold = max(dist_threshold, min_dist)
    slice_diff = np.abs(pred_z - gt_z)
    # TP if predicted center within radius and slice within slice offset
    return dist <= dist_threshold and slice_diff <= slice_offset


def froc_curve(df_gt, df_pred, thresholds, verbose, cases="all", per_side=False):
    if per_side:
        return froc_curve_per_side(df_gt, df_pred, thresholds, verbose, cases)

    assert cases in ["all", "cancer", "benign"]

    if not cases == "all":
        df_exclude = df_gt[~(df_gt["Class"] == cases)]
        df_gt = df_gt[df_gt["Class"] == cases]
        df_pred = df_pred[~(df_pred["StudyUID"].isin(set(df_exclude["StudyUID"])))]

    total_volumes = len(df_pred.drop_duplicates(subset=["StudyUID", "View"]))
    total_tps = len(df_gt)

    tpr = []
    fps = []

    if verbose:
        print("{} cases FROC:".format(cases.upper()))

    for th in sorted(thresholds, reverse=True):
        df_th = df_pred[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(
            subset=["StudyUID", "View", "TP", "GTID"]
        )
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_tps
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_volumes
        tpr.append(tpr_th)
        fps.append(fps_th)

        if verbose:
            print(
                "Sensitivity {0:.2f} at {1:.2f} FPs/volume (threshold: {2:.4f})".format(
                    tpr_th * 100, fps_th, th
                )
            )

    return tpr, fps


def froc_curve_per_side(df_gt, df_pred, thresholds, verbose, cases="all"):
    """
    Compute FROC curve per side/breast. All lesions in a breast are considered TP if
    any lesion in that breast is detected.
    """
    assert cases in ["all", "cancer", "benign"]

    if not cases == "all":
        df_exclude = df_gt[~(df_gt["Class"] == cases)]
        df_gt = df_gt[df_gt["Class"] == cases]
        df_pred = df_pred[~(df_pred["StudyUID"].isin(set(df_exclude["StudyUID"])))]

    df_gt["Side"] = df_gt["View"].astype(str).str[0]
    df_pred["Side"] = df_pred["View"].astype(str).str[0]

    total_volumes = len(df_pred.drop_duplicates(subset=["StudyUID", "View"]))
    total_tps = len(df_gt.drop_duplicates(subset=["PatientID", "Side"]))

    tpr = []
    fps = []

    if verbose:
        print("{} cases FROC:".format(cases.upper()))

    for th in sorted(thresholds, reverse=True):
        df_th = df_pred[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(subset=["PatientID", "Side", "TP"])
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_tps
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_volumes
        tpr.append(tpr_th)
        fps.append(fps_th)

        if verbose:
            print(
                "Sensitivity {0:.2f} at {1:.2f} FPs/volume (threshold: {2:.4f})".format(
                    tpr_th * 100, fps_th, th
                )
            )

    return tpr, fps


def froc_curve_per_patient(df_gt, df_pred, thresholds, verbose, cases="all"):
    assert cases in ["all", "cancer", "benign"]

    if not cases == "all":
        df_exclude = df_gt[~(df_gt["Class"] == cases)]
        df_gt = df_gt[df_gt["Class"] == cases]
        df_pred = df_pred[~(df_pred["StudyUID"].isin(set(df_exclude["StudyUID"])))]

    total_volumes = len(df_pred.drop_duplicates(subset=["StudyUID", "View"]))
    df_gt["Side"] = df_gt["View"]
    total_tps = len(set(df_gt["PatientID"]))

    tpr = []
    fps = []

    if verbose:
        print("{} cases FROC:".format(cases.upper()))

    for th in sorted(thresholds, reverse=True):
        df_th = df_pred[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(subset=["PatientID", "TP"])
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_tps
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_volumes
        tpr.append(tpr_th)
        fps.append(fps_th)

        if verbose:
            print(
                "Sensitivity {0:.2f} at {1:.2f} FPs/volume (threshold: {2:.4f})".format(
                    tpr_th * 100, fps_th, th
                )
            )

    return tpr, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py")
    parser.add_argument(
        "--gt",
        help="csv file with ground truth bounding boxes",
        default="/data/bboxes_v2.csv",
    )
    parser.add_argument(
        "--predictions", help="CSV file with predictions", required=True
    )
    parser.add_argument(
        "--figure",
        help="figure image file name (destination folder must exist)",
        default="./froc.png",
    )
    parser.add_argument(
        "--per-side",
        default=False,
        action="store_true",
        help="flag to compute FROC curve based on side/breast and not boxes",
    )
    parser.add_argument(
        "--two-dim",
        default=False,
        action="store_true",
        help="flag to evaluate as 2d by making GT boxes span entire volume",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="flag to print values of points for FROC curve and prediction thresholds",
    )
    args = parser.parse_args()
    main(args)
