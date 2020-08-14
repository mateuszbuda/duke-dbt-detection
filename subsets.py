import numpy as np
import pandas as pd
import random

num_cancers_valid = 20
num_benigns_valid = 20
num_actions_valid = 40
num_normals_valid = 200


def data_frame_subset(csv_views, csv_boxes, subset, seed=42):
    subset = str(subset).lower()
    assert subset in ["test", "train", "validation", "all"]
    df_all = pd.read_csv(
        csv_views, dtype={
            "Normal": np.int, "Actionable": np.int, "Benign": np.int, "Cancer": np.int
        }
    )

    if subset in ["test", "all"]:
        return df_all

    df_box = pd.read_csv(csv_boxes)
    df_box = df_box[df_box["PatientID"].isin(df_all["PatientID"])]
    df_box["Diag"] = np.sqrt((df_box["Width"] ** 2 + df_box["Height"] ** 2))

    cancer_ids = sorted(set(df_box[df_box["Class"] == "cancer"]["PatientID"]))
    cancer_ad_ids = sorted(
        set(df_box[(df_box["Class"] == "cancer") & (df_box["AD"] == 1)]["PatientID"])
    )
    cancer_ms_ids = sorted(set(cancer_ids) - set(cancer_ad_ids))

    benign_ids = sorted(set(df_box[df_box["Class"] == "benign"]["PatientID"]))
    benign_ad_ids = sorted(
        set(df_box[(df_box["Class"] == "benign") & (df_box["AD"] == 1)]["PatientID"])
    )
    benign_ms_ids = sorted(set(benign_ids) - set(benign_ad_ids))

    action_ids = set(df_all[df_all["Actionable"] == 1]["PatientID"])
    action_ids = sorted(list(action_ids))

    normal_ids = sorted(set(df_all[df_all["Normal"] == 1]["PatientID"]))

    # get prevalence of ADs
    n_boxes = float(len(df_box))
    ads_cancer_ratio = sum(df_box[df_box["Class"] == "cancer"]["AD"]) / n_boxes
    ads_benign_ratio = sum(df_box[df_box["Class"] == "benign"]["AD"]) / n_boxes

    random.seed(seed)
    validation_ids = []

    # sample normals
    valid_normal_ids = random.sample(normal_ids, num_normals_valid)
    validation_ids.extend(valid_normal_ids)

    # sample actionables
    # get the number of actionables already sampled as normals
    num_actions_in_valid_normals = len(set(valid_normal_ids).intersection(set(action_ids)))
    valid_action_ids = random.sample(
        set(action_ids).difference(normal_ids),
        num_actions_valid - num_actions_in_valid_normals
    )
    validation_ids.extend(valid_action_ids)

    # sample cancer ADs
    num_cancers_ads_valid = int(np.round(ads_cancer_ratio * num_cancers_valid))
    validation_ids.extend(
        random.sample(cancer_ad_ids, num_cancers_ads_valid)
    )
    # sample cancer masses
    num_cancers_mas_valid = num_cancers_valid - num_cancers_ads_valid
    validation_ids.extend(
        random.sample(cancer_ms_ids, num_cancers_mas_valid)
    )

    # sample benign ADs
    num_benigns_ads_valid = int(np.round(ads_benign_ratio * num_benigns_valid))
    validation_ids.extend(
        random.sample(benign_ad_ids, num_benigns_ads_valid)
    )
    # sample benign masses
    num_benigns_mas_valid = num_benigns_valid - num_benigns_ads_valid
    validation_ids.extend(
        random.sample(benign_ms_ids, num_benigns_mas_valid)
    )

    if subset == "validation":
        return df_all[df_all["PatientID"].isin(validation_ids)]

    return df_all[~df_all["PatientID"].isin(validation_ids)]


if __name__ == "__main__":
    for subset in ["all", "train", "validation"]:
        print(subset.upper() + ":")
        df = data_frame_subset(
            "/data/data_train_v2.csv", "/data/bboxes_v2.csv", subset, seed=42
        )
        print("Volumes: {}".format(len(df)))
        print("Studies: {}".format(len(set(df["StudyUID"]))))
        print("Patients: {}".format(len(set(df["PatientID"]))))
        print("Cancer volumes: {}".format(len(df[df["Cancer"] == 1])))
        print("Cancer studies: {}".format(len(set(df[df["Cancer"] == 1]["StudyUID"]))))
        print("Cancer patients: {}".format(len(set(df[df["Cancer"] == 1]["PatientID"]))))
        print("Benign volumes: {}".format(len(df[(df["Benign"] == 1)])))
        print("Benign studies: {}".format(len(set(df[(df["Benign"] == 1)]["StudyUID"]))))
        print("Benign patients: {}".format(len(set(df[(df["Benign"] == 1)]["PatientID"]))))
        print("Actionable volumes: {}".format(len(df[(df["Actionable"] == 1)])))
        print("Actionable studies: {}".format(len(set(df[(df["Actionable"] == 1)]["StudyUID"]))))
        print("Actionable patients: {}".format(len(set(df[(df["Actionable"] == 1)]["PatientID"]))))
        print("Normal volumes: {}".format(len(df[df["Normal"] == 1])))
        print("Normal studies: {}".format(len(set(df[df["Normal"] == 1]["StudyUID"]))))
        print("Normal patients: {}".format(len(set(df[df["Normal"] == 1]["PatientID"]))))
