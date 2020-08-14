import argparse

import pandas as pd
from tqdm import tqdm

from utils import iou_3d, box_union_3d


def main(args):
    preds = pd.read_csv(args.predictions)
    volumes = preds.drop_duplicates(subset=["PatientID", "StudyUID", "View"])
    volumes = volumes[["PatientID", "StudyUID", "View"]]
    df = pd.DataFrame()

    for index, row in tqdm(volumes.iterrows(), total=len(volumes)):
        volume_preds = preds[
            (preds["PatientID"] == row["PatientID"])
            & (preds["StudyUID"] == row["StudyUID"])
            & (preds["View"] == row["View"])
        ]

        volume_preds = merge_boxes(
            volume_preds.copy(),
            min_overlap=args.nms_threshold,
            max_pred_ratio=args.max_pred_ratio,
        )

        df = df.append(volume_preds, ignore_index=True, sort=False)

    df = df[
        ["PatientID", "StudyUID", "View", "Score", "Z", "X", "Y", "Depth", "Width", "Height"]
    ]
    df.to_csv(args.output, index=False)


def merge_boxes(df, min_overlap=0.5, max_pred_ratio=10.0):
    df["X1"] = df["X"] + df["Width"]
    df["Y1"] = df["Y"] + df["Height"]
    df["Z1"] = df["Z"] + df["Depth"]
    columns = ["X", "Y", "Z", "X1", "Y1", "Z1", "Score"]
    boxes = df[columns].values.tolist()

    did_merge = True
    while did_merge:
        did_merge = False
        for i in range(len(boxes)):
            A = boxes[i]
            for j in range(i + 1, len(boxes)):
                B = boxes[j]
                if max(A[6], B[6]) / min(A[6], B[6]) <= max_pred_ratio:
                    if iou_3d(A, B) > min_overlap:
                        boxes[i] = box_union_3d(A, B)
                        del boxes[j]
                        did_merge = True
                        break
            if did_merge:
                break

    df_nms = pd.DataFrame(boxes, columns=columns)
    df_nms["Width"] = df_nms["X1"] - df_nms["X"]
    df_nms["Height"] = df_nms["Y1"] - df_nms["Y"]
    df_nms["Depth"] = df_nms["Z1"] - df_nms["Z"]
    df_nms.drop(columns=["X1", "Y1", "Z1"], inplace=True)
    df_nms["PatientID"] = df["PatientID"].iloc[0]
    df_nms["StudyUID"] = df["StudyUID"].iloc[0]
    df_nms["View"] = df["View"].iloc[0]
    return df_nms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocessing of predicted bounding boxes"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="File path with predicted bounding boxes",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to postprocessed csv output file",
    )
    parser.add_argument(
        "--nms-threshold",
        default=0.5,
        help="Threshold for minimum overlap in NMS post-processing",
    )
    parser.add_argument(
        "--max-pred-ratio",
        type=float,
        default=10.0,
        help="Threshold for maximum ratio between confidence score of predicted boxes in NMS post-processing",
    )
    args = parser.parse_args()
    main(args)
