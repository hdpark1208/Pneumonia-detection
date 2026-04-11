import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

import torchvision
from torchvision.models import ResNet50_Weights


# def get_model(num_classes=2):
#     model = torchvision.models.detection.retinanet_resnet50_fpn(
#         weights=None,
#         weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
#         num_classes=num_classes,
#     )
#     return model