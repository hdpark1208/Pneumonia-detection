from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from utils import calculate_area


class PneumoniaDataset(Dataset):
    def __init__(self, records, transforms=None):
        self.records = records
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        boxes = record["boxes"]
        labels = record["labels"]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": calculate_area(boxes),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class PneumoniaTestDataset(Dataset):
    def __init__(self, records, transforms=None):
        self.records = records
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, record["patientId"]