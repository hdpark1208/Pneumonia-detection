from pathlib import Path
from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PneumoniaDataset
from model import get_model
from utils import (
    collate_fn,
    make_patient_records,
    rebalance_train_records,
    save_json,
    seed_everything,
    train_valid_split,
)


BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")
TRAIN_META = BASE_DIR / "stage2_train_metadata.csv"
IMAGE_DIR = BASE_DIR / "Training" / "Images"

EXPERIMENTS_ROOT = Path("experiments")
EXPERIMENTS_ROOT.mkdir(exist_ok=True)


EXPERIMENT_CONFIGS = [
    {
        "name": "fstrRC_SGD_lr005_neg15",
        "optimizer": "SGD",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "neg_ratio": 1.5,
        "batch_size": 4,
        "num_epochs": 7,
        "step_size": 3,
        "gamma": 0.1,
        "seed": 42,
    },    
]

def evaluate_loss(model, dataloader, device):
    model.train()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, epoch, train_loss, valid_loss, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        save_path,
    )


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]

    if cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
    elif cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

    elif cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"지원하지 않는 optimizer: {cfg['optimizer']}")

    return optimizer


def run_experiment(cfg, base_records):
    print("\n" + "=" * 100)
    print(f"실험 시작: {cfg['name']}")
    print("=" * 100)

    seed_everything(cfg["seed"])

    experiment_dir = EXPERIMENTS_ROOT / cfg["name"]
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, experiment_dir / "config.json")

    train_records, valid_records = train_valid_split(base_records, valid_ratio=0.2, seed=cfg["seed"])

    print(f"분할 전 train: {len(train_records)}")
    print(f"valid: {len(valid_records)}")

    train_records = rebalance_train_records(train_records, neg_ratio=cfg["neg_ratio"], seed=cfg["seed"])

    print(f"다운샘플링 후 train: {len(train_records)}")

    train_dataset = PneumoniaDataset(train_records)
    valid_dataset = PneumoniaDataset(valid_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))

    model = get_model(num_classes=2)
    model.to(device)

    optimizer = build_optimizer(model, cfg)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["step_size"],
        gamma=cfg["gamma"],
    )

    best_valid_loss = float("inf")
    history = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            if epoch == 0 and step == 1:
                print("first image device:", images[0].device)
                print("first target boxes device:", targets[0]["boxes"].device)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

            if step % 200 == 0:
                print(f"[{cfg['name']}] [Epoch {epoch + 1} | Step {step}] loss = {loss.item():.4f}")

        train_loss_avg = train_loss_sum / max(train_steps, 1)
        valid_loss_avg = evaluate_loss(model, valid_loader, device)

        print(f"\n[{cfg['name']}] Epoch {epoch + 1}/{cfg['num_epochs']}")
        print(f"Train Loss: {train_loss_avg:.4f}")
        print(f"Valid Loss: {valid_loss_avg:.4f}\n")

        epoch_ckpt_path = checkpoints_dir / f"epoch_{epoch + 1}.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            train_loss=train_loss_avg,
            valid_loss=valid_loss_avg,
            save_path=epoch_ckpt_path,
        )

        if valid_loss_avg < best_valid_loss:
            best_valid_loss = valid_loss_avg
            best_model_path = experiment_dir / "best_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss_avg,
                valid_loss=valid_loss_avg,
                save_path=best_model_path,
            )
            print(f"best model 갱신: {best_model_path}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_avg),
                "valid_loss": float(valid_loss_avg),
            }
        )
        save_json(history, experiment_dir / "history.json")

        lr_scheduler.step()

    summary = {
        "experiment_name": cfg["name"],
        "best_valid_loss": float(best_valid_loss),
        "config": deepcopy(cfg),
    }
    save_json(summary, experiment_dir / "summary.json")

    print(f"실험 종료: {cfg['name']}")
    print(f"best_valid_loss: {best_valid_loss:.4f}")


def main():
    first_seed = EXPERIMENT_CONFIGS[0]["seed"]
    seed_everything(first_seed)

    df = pd.read_csv(TRAIN_META)
    base_records = make_patient_records(df, IMAGE_DIR)
    print(f"전체 이미지 수: {len(base_records)}")

    for cfg in EXPERIMENT_CONFIGS:
        run_experiment(cfg, base_records)


if __name__ == "__main__":
    main()