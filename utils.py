from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def make_patient_records(df: pd.DataFrame, image_dir: Path):
    records = []

    for patient_id, group in df.groupby("patientId"):
        image_path = image_dir / f"{patient_id}.png"

        boxes = []

        for _, row in group.iterrows():
            if int(row["Target"]) == 1:
                x1 = float(row["x"])
                y1 = float(row["y"])
                x2 = x1 + float(row["width"])
                y2 = y1 + float(row["height"])
                boxes.append([x1, y1, x2, y2])

        records.append(
            {
                "patientId": patient_id,
                "image_path": str(image_path),
                "boxes": boxes,
                "labels": [1] * len(boxes),
            }
        )

    return records


def train_valid_split(records, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    split = int(len(indices) * (1 - valid_ratio))
    train_idx = indices[:split]
    valid_idx = indices[split:]

    train_records = [records[i] for i in train_idx]
    valid_records = [records[i] for i in valid_idx]

    return train_records, valid_records


def calculate_area(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return widths * heights


def make_test_records(test_df: pd.DataFrame, image_dir: Path):
    records = []

    patient_ids = test_df["patientId"].unique().tolist()

    for patient_id in patient_ids:
        image_path = image_dir / f"{patient_id}.png"
        records.append(
            {
                "patientId": patient_id,
                "image_path": str(image_path),
            }
        )

    return records


def rebalance_train_records(train_records, neg_ratio=1.5, seed=42):
    rng = random.Random(seed)

    pos_records = [r for r in train_records if len(r["boxes"]) > 0]
    neg_records = [r for r in train_records if len(r["boxes"]) == 0]

    rng.shuffle(neg_records)
    keep_neg = int(len(pos_records) * neg_ratio)
    neg_records = neg_records[:keep_neg]

    balanced_records = pos_records + neg_records
    rng.shuffle(balanced_records)

    return balanced_records


def prediction_to_string(
    boxes,
    scores,
    score_threshold=0.25,
    top_k=3,
    min_area=0,
):
    pairs = []

    for box, score in zip(boxes, scores):
        score = float(score)
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area < min_area:
            continue

        pairs.append((score, [x1, y1, x2, y2]))

    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:top_k]

    prediction_parts = []

    for score, (x1, y1, x2, y2) in pairs:
        width = x2 - x1
        height = y2 - y1

        prediction_parts.extend([
            f"{score:.4f}",
            f"{x1:.1f}",
            f"{y1:.1f}",
            f"{width:.1f}",
            f"{height:.1f}",
        ])

    return " ".join(prediction_parts)


def save_json(data, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)