import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
import sys
sys.path.append("/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/EfficientSAM")
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building MobileSAM predictor
HQSAM_CHECKPOINT_PATH = "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/EfficientSAM/sam_hq_vit_tiny.pth"
checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
light_hqsam = setup_model()
light_hqsam.load_state_dict(checkpoint, strict=True)
light_hqsam.to(device=DEVICE)

sam_predictor = SamPredictor(light_hqsam)


# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/outputs/desk_top3.jpg"
CLASSES = ["keyboard", "ruler", "bottle", "cube", "wireless controller", "scissors", "paper"]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# load image
image = cv2.imread(SOURCE_IMAGE_PATH)
image_H, image_W = image.shape[:2]
# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# save the annotated grounding dino image
cv2.imwrite("EfficientSAM/LightHQSAM/groundingdino_annotated_image.jpg", annotated_frame)


# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    NMS_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

print("detected classes:", detections)
class Args:
    def __init__(self):
        self.track_thresh = 0.2
        self.track_buffer = 30
        self.match_thresh = 0.5
        self.mot20 = False

args = Args()
tracker = BYTETracker(track_thresh = 0.2, track_buffer = 30, match_thresh = 0.5, mot20 = False)

confidence_reshaped = detections.confidence[:, np.newaxis]
dets = np.hstack((detections.xyxy, confidence_reshaped))
print("dets:", dets)

online_targets = tracker.update(dets, [image_H, image_W], [image_H, image_W])
print("online_targets:", online_targets)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False,
            hq_token_only=True,
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
cv2.imwrite("/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/outputs/rounded_light_hqsam_annotated_image111.jpg", annotated_image)
