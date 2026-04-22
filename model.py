import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights

def get_model(num_classes):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=0.25,
        box_nms_thresh=0.30,
        box_detections_per_img=5,
        rpn_nms_thresh=0.30,   # 필요 시 더 낮게
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# def get_model(num_classes=2):
#     model = torchvision.models.detection.retinanet_resnet50_fpn(
#         weights=None,
#         weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
#         num_classes=num_classes,
#     )
#     return model