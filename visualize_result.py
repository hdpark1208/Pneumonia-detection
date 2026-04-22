from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch

from dataset import PneumoniaDataset
from model import get_model
from utils import make_patient_records, train_valid_split, seed_everything
from torchvision.ops import nms


BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")
TRAIN_META = BASE_DIR / "stage2_train_metadata.csv"
TRAIN_IMAGE_DIR = BASE_DIR / "Training" / "Images"

# 여기만 네 상황에 맞게 바꿔
MODEL_PATH = Path("experiments/fstrRC_SGD_lr005_neg15/best_model.pth")
# 예: experiments/exp01_sgd_lr0005_neg15/best_model.pth

OUTPUT_DIR = Path("vis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def tensor_to_image(image: torch.Tensor):
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return image

# new prediction for NMS
def filter_prediction(
    prediction,
    score_threshold=0.25,
    top_k=5,
    min_area=0,
    nms_threshold=0.3,
):
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    # 1) score threshold
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return [], []

    # 2) min_area filtering
    keep_indices = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area >= min_area:
            keep_indices.append(i)

    if len(keep_indices) == 0:
        return [], []

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]

    # 3) additional NMS
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return [], []

    # 4) top_k
    order = torch.argsort(scores, descending=True)
    order = order[:top_k]
    boxes = boxes[order]
    scores = scores[order]

    return boxes.tolist(), scores.tolist()

# def filter_prediction(prediction, score_threshold=0.25, top_k=5, min_area=0):
#     boxes = prediction["boxes"].detach().cpu().numpy()
#     scores = prediction["scores"].detach().cpu().numpy()

#     pairs = []
#     for box, score in zip(boxes, scores):
#         if float(score) < score_threshold:
#             continue

#         x1, y1, x2, y2 = box.tolist()
#         area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
#         if area < min_area:
#             continue

#         pairs.append((float(score), [x1, y1, x2, y2]))

#     pairs.sort(key=lambda x: x[0], reverse=True)
#     pairs = pairs[:top_k]

#     filtered_boxes = [box for score, box in pairs]
#     filtered_scores = [score for score, box in pairs]

#     return filtered_boxes, filtered_scores


def draw_gt(ax, gt_boxes):
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect)


def draw_pred(ax, pred_boxes, pred_scores):
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}")


def save_gt_only(image, target, save_path, title="GT"):
    image_np = tensor_to_image(image)
    gt_boxes = target["boxes"].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    draw_gt(ax, gt_boxes)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_pred_only(
    image,
    prediction,
    save_path,
    score_threshold=0.25,
    top_k=5,
    min_area=0,
    nms_threshold=0.3,
    title="Prediction",
):
    image_np = tensor_to_image(image)
    pred_boxes, pred_scores = filter_prediction(
        prediction,
        score_threshold=score_threshold,
        top_k=top_k,
        min_area=min_area,
        nms_threshold=nms_threshold,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    draw_pred(ax, pred_boxes, pred_scores)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_gt_and_pred(image, target, prediction, save_path, score_threshold=0.25, top_k=5, min_area=0, title="GT vs Prediction"):
    image_np = tensor_to_image(image)
    gt_boxes = target["boxes"].detach().cpu().numpy()
    pred_boxes, pred_scores = filter_prediction(
        prediction,
        score_threshold=score_threshold,
        top_k=top_k,
        min_area=min_area,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    draw_gt(ax, gt_boxes)
    draw_pred(ax, pred_boxes, pred_scores)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def load_model_cpu(model_path: Path):
    device = torch.device("cpu")  # 학습 중일 때 안전하게 CPU 사용
    checkpoint = torch.load(model_path, map_location=device)

    model = get_model(num_classes=2)

    # checkpoint 구조가 dict일 수도 있고 state_dict만 있을 수도 있음
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device


def main():
    seed_everything(42)

    df = pd.read_csv(TRAIN_META)
    records = make_patient_records(df, TRAIN_IMAGE_DIR)
    train_records, valid_records = train_valid_split(records, valid_ratio=0.2, seed=42)
    valid_dataset = PneumoniaDataset(valid_records)

    model, device = load_model_cpu(MODEL_PATH)

    # bbox 있는 샘플 / 없는 샘플 따로 찾기
    pos_indices = [i for i in range(len(valid_dataset)) if valid_dataset.records[i]["boxes"]]
    neg_indices = [i for i in range(len(valid_dataset)) if not valid_dataset.records[i]["boxes"]]

    rng = random.Random(42)

    chosen_pos = rng.sample(pos_indices, k=min(10, len(pos_indices)))
    chosen_neg = rng.sample(neg_indices, k=min(2, len(neg_indices)))

    chosen_indices = chosen_pos + chosen_neg

    print("시각화 대상 인덱스:", chosen_indices)

    for idx in chosen_indices:
        image, target = valid_dataset[idx]

        with torch.no_grad():
            prediction = model([image.to(device)])[0]

        patient_id = valid_dataset.records[idx]["patientId"]

        save_gt_only(
            image,
            target,
            OUTPUT_DIR / f"{patient_id}_gt.png",
            title=f"GT - {patient_id}",
        )

        save_pred_only(
            image,
            prediction,
            OUTPUT_DIR / f"{patient_id}_pred_before_nms.png",
            score_threshold=0.25,
            top_k=5,
            min_area=0,
            nms_threshold=1.0,   # 사실상 NMS 거의 안 하는 수준
            title=f"Prediction Before NMS - {patient_id}",
        )

        # NMS 적용 prediction 저장
        save_pred_only(
            image,
            prediction,
            OUTPUT_DIR / f"{patient_id}_pred_after_nms.png",
            score_threshold=0.25,
            top_k=5,
            min_area=0,
            nms_threshold=0.3,
            title=f"Prediction After NMS - {patient_id}",
        )

        # save_gt_and_pred(
        #     image,
        #     target,
        #     prediction,
        #     OUTPUT_DIR / f"{patient_id}_gt_pred.png",
        #     score_threshold=0.25,
        #     top_k=5,
        #     min_area=0,
        #     title=f"GT vs Prediction - {patient_id}",
        # )


    print(f"\n모든 이미지 저장 완료: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()