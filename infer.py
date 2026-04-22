from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from dataset import PneumoniaTestDataset
from model import get_model
from utils import (
    collate_fn,
    make_test_records,
    seed_everything,
)


BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")
TEST_META = BASE_DIR / "stage2_test_metadata.csv"
TEST_IMAGE_DIR = BASE_DIR / "Test"

# 여기 네 모델 경로에 맞게 수정
MODEL_PATH = Path("experiments/fstrRC_SGD_lr005_neg15/best_model.pth")

SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(exist_ok=True)


def postprocess_prediction(
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


def prediction_to_string(boxes, scores):
    if len(boxes) == 0:
        return ""

    parts = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        parts.extend([
            f"{score:.4f}",
            f"{x1:.1f}",
            f"{y1:.1f}",
            f"{width:.1f}",
            f"{height:.1f}",
        ])

    return " ".join(parts)


def run_inference(
    score_threshold=0.25,
    top_k=5,
    min_area=0,
    nms_threshold=0.3,
    submission_name="submission.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = get_model(num_classes=2)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    test_df = pd.read_csv(TEST_META)
    print("test_df shape:", test_df.shape)
    print("test_df columns:", test_df.columns.tolist())

    test_records = make_test_records(test_df, TEST_IMAGE_DIR)
    print("test image count:", len(test_records))

    test_dataset = PneumoniaTestDataset(test_records)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    submission_rows = []

    with torch.no_grad():
        for batch_idx, (images, patient_ids) in enumerate(test_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            predictions = model(images)

            for patient_id, prediction in zip(patient_ids, predictions):
                boxes, scores = postprocess_prediction(
                    prediction,
                    score_threshold=score_threshold,
                    top_k=top_k,
                    min_area=min_area,
                    nms_threshold=nms_threshold,
                )

                pred_str = prediction_to_string(boxes, scores)

                submission_rows.append(
                    {
                        "patientId": patient_id,
                        "PredictionString": pred_str,
                    }
                )

            if batch_idx % 100 == 0:
                print(f"inference batch {batch_idx} 완료")

    submission_df = pd.DataFrame(submission_rows)
    submission_path = SUBMISSION_DIR / submission_name
    submission_df.to_csv(submission_path, index=False)

    print(f"submission 저장 완료: {submission_path}")
    print(submission_df.head())

    return submission_path


def main():
    seed_everything(42)

    settings = [
        {
            "score_threshold": 0.25,
            "top_k": 5,
            "min_area": 0,
            "nms_threshold": 0.30,
            "name": "submission_thr025_top5_nms030.csv",
        },
        {
            "score_threshold": 0.20,
            "top_k": 5,
            "min_area": 0,
            "nms_threshold": 0.30,
            "name": "submission_thr020_top5_nms030.csv",
        },
        {
            "score_threshold": 0.25,
            "top_k": 5,
            "min_area": 0,
            "nms_threshold": 0.20,
            "name": "submission_thr025_top5_nms020.csv",
        },
    ]

    for cfg in settings:
        print("\n" + "=" * 80)
        print(
            f"score_threshold={cfg['score_threshold']}, "
            f"top_k={cfg['top_k']}, "
            f"min_area={cfg['min_area']}, "
            f"nms_threshold={cfg['nms_threshold']}"
        )
        print("=" * 80)

        run_inference(
            score_threshold=cfg["score_threshold"],
            top_k=cfg["top_k"],
            min_area=cfg["min_area"],
            nms_threshold=cfg["nms_threshold"],
            submission_name=cfg["name"],
        )


if __name__ == "__main__":
    main()