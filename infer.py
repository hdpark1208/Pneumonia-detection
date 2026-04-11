from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PneumoniaTestDataset
from model import get_model
from utils import (
    collate_fn,
    make_test_records,
    prediction_to_string,
    seed_everything,
)


BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")
TEST_META = BASE_DIR / "stage2_test_metadata.csv"
TEST_IMAGE_DIR = BASE_DIR / "Test"

EXPERIMENT_NAME = "exp02_sgd_lr0003_neg15"
MODEL_PATH = Path("experiments") / EXPERIMENT_NAME / "best_model.pth"

SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(exist_ok=True)


def run_inference(score_threshold=0.25, top_k=5, min_area=0, submission_name="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = get_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
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
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    submission_rows = []

    with torch.no_grad():
        for batch_idx, (images, patient_ids) in enumerate(test_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            predictions = model(images)

            for patient_id, prediction in zip(patient_ids, predictions):
                pred_str = prediction_to_string(
                    prediction["boxes"].detach().cpu(),
                    prediction["scores"].detach().cpu(),
                    score_threshold=score_threshold,
                    top_k=top_k,
                    min_area=min_area,
                )

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


def main():
    seed_everything(42)

    run_inference(
        score_threshold=0.25,
        top_k=5,
        min_area=0,
        submission_name=f"{EXPERIMENT_NAME}_thr025_top5.csv",
    )


if __name__ == "__main__":
    main()