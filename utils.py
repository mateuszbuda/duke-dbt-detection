import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale

from dataset import TomoDetectionDataset


def log_images(x, y_true, y_pred):
    images = []
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    for c in range(y_true_np.shape[0]):
        pred_bboxes = label2bboxes(y_pred_np[c])
        gt_bboxes = label2bboxes(y_true_np[c], n_boxes=np.sum(y_true_np[c] == 1))
        image = np.squeeze(x_np[c, 0])
        image -= np.min(image)
        image /= np.max(image)
        image_bboxes = draw_predictions(
            image, pred_bboxes, gt_bboxes
        )
        images.append(image_bboxes)
    return images


def label2bboxes(label, n_boxes=6, min_size=28):
    obj = label[0]
    loc = label[1:]
    th = sorted(obj.flatten(), reverse=True)[n_boxes]
    bboxes = {"X": [], "Y": [], "Width": [], "Height": [], "Score": []}
    csz = TomoDetectionDataset.cell_size
    anchor = TomoDetectionDataset.anchor
    for i in range(obj.shape[0]):
        for j in range(obj.shape[1]):
            if obj[i, j] > th:
                y_cell = i * csz + csz / 2
                x_cell = j * csz + csz / 2
                y_center = y_cell + (csz / 2) * loc[0, i, j]
                x_center = x_cell + (csz / 2) * loc[1, i, j]
                h = anchor[0] * loc[2, i, j] ** 2
                w = anchor[1] * loc[3, i, j] ** 2
                if obj[i, j] == 1:
                    h = max(h, min_size)
                    w = max(w, min_size)
                bboxes["Y"].append(max(0, y_center - (h / 2)))
                bboxes["X"].append(max(0, x_center - (w / 2)))
                bboxes["Width"].append(w)
                bboxes["Height"].append(h)
                bboxes["Score"].append(obj[i, j])
    return bboxes


def draw_predictions(image, pred_boxes, gt_boxes):
    image = np.stack((image,) * 3, axis=-1)
    red = [np.max(image), 0, 0]
    green = [0, np.max(image), 0]
    for i in range(len(gt_boxes["X"])):
        x, y = int(gt_boxes["X"][i]), int(gt_boxes["Y"][i])
        w, h = int(gt_boxes["Width"][i]), int(gt_boxes["Height"][i])
        image = draw_bbox(image, x, y, w, h, c=green, lw=4)
    boxes = zip(pred_boxes["X"], pred_boxes["Y"], pred_boxes["Width"], pred_boxes["Height"], pred_boxes["Score"])
    for box in sorted(boxes, key=lambda a: a[-1]):
        x, y = int(box[0]), int(box[1])
        x, y = max(x, 0), max(y, 0)
        w, h = int(box[2]), int(box[3])
        image = draw_bbox(image, x, y, w, h, c=red, lw=3)
        image = draw_score(image, box[-1], x, y)
    return image


def draw_bbox(img, x, y, w, h, c=None, lw=4):
    x = min(max(x, 0), img.shape[1] - 1)
    y = min(max(y, 0), img.shape[0] - 1)
    if c is None:
        c = np.max(img)
        if len(img.shape) > 2:
            c = [c] + [0] * (img.shape[-1] - 1)
    img[y : y + lw, x : x + w] = c
    img[y + h - lw : y + h, x : x + w] = c
    img[y : y + h, x : x + lw] = c
    img[y : y + h, x + w - lw : x + w] = c
    return img


def draw_score(img, score, x, y):
    score = int(min(max(0, score * 100), 100))
    txt_img = text_image(str(score) + "%") * np.max(img)
    txt_h, txt_w = txt_img.shape[0], txt_img.shape[1]
    if y + txt_h > img.shape[0]:
        max_h = img.shape[0] - y
        txt_img = txt_img[:max_h]
    if x + txt_w > img.shape[1]:
        max_w = img.shape[1] - x
        txt_img = txt_img[:, :max_w]
    if img[y : y + txt_h, x : x + txt_w].shape == txt_img.shape:
        img[y : y + txt_h, x : x + txt_w] = txt_img
    return img


def text_image(text, bg=(255, 0, 0), margin=4):
    bg = tuple([255 - c for c in bg])
    margin = margin // 2
    font = ImageFont.load_default()
    text_width, text_height = font.getsize(text)
    canvas = Image.new("RGB", [text_width + 2 * margin - 1, text_height], bg)
    draw = ImageDraw.Draw(canvas)
    offset = (margin, 0)
    black = "#FFFFFF"
    draw.text(offset, text, font=font, fill=black)
    image = (255 - np.asarray(canvas)) / 255.0
    return rescale(
        image,
        2.0,
        anti_aliasing=False,
        preserve_range=True,
        multichannel=True,
        mode="edge",
    )


def iou_3d(A, B):
    x0a, y0a, z0a, x1a, y1a, z1a = A[0], A[1], A[2], A[3], A[4], A[5]
    x0b, y0b, z0b, x1b, y1b, z1b = B[0], B[1], B[2], B[3], B[4], B[5]
    x0i, x1i = max(x0a, x0b), min(x1a, x1b)
    y0i, y1i = max(y0a, y0b), min(y1a, y1b)
    z0i, z1i = max(z0a, z0b), min(z1a, z1b)
    wi = x1i - x0i
    if wi <= 0:
        return 0.0
    hi = y1i - y0i
    if hi <= 0:
        return 0.0
    di = z1i - z0i
    if di <= 0:
        return 0.0
    area_a = (x1a - x0a) * (y1a - y0a) * (z1a - z0a)
    area_b = (x1b - x0b) * (y1b - y0b) * (z1b - z0b)
    intersection = (x1i - x0i) * (y1i - y0i) * (z1i - z0i)
    union = area_a + area_b - intersection
    return float(intersection) / union


def box_union_3d(A, B):
    x0 = min(A[0], B[0])
    y0 = min(A[1], B[1])
    z0 = min(A[2], B[2])
    x1 = max(A[3], B[3])
    y1 = max(A[4], B[4])
    z1 = max(A[5], B[5])
    score = max(A[6], B[6])
    return [x0, y0, z0, x1, y1, z1, score]
