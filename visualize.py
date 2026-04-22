import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch



def tensor_to_image(image: torch.Tensor):
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return image


def show_sample(image, target, figsize=(8, 8), title="Sample with GT Boxes"):
    image_np = tensor_to_image(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    boxes = target["boxes"].detach().cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)

    plt.show()


def show_prediction(image, prediction, score_threshold=0.3, figsize=(8, 8), title="Prediction"):
    image_np = tensor_to_image(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue

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

    plt.show()


def show_gt_and_prediction(image, target, prediction, score_threshold=0.3, figsize=(10, 10), title="GT and Prediction"):
    image_np = tensor_to_image(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    gt_boxes = target["boxes"].detach().cpu().numpy()
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

    pred_boxes = prediction["boxes"].detach().cpu().numpy()
    pred_scores = prediction["scores"].detach().cpu().numpy()

    for box, score in zip(pred_boxes, pred_scores):
        if score < score_threshold:
            continue

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

    plt.show()