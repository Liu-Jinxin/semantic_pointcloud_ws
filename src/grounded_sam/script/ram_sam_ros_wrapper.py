#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
import argparse
import os
import numpy as np
import json
import torch
import torchvision
import matplotlib.cm as cm
from PIL import Image as PILImage

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
sys.path.append('/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS


class GroundedSAMServiceNode:
    def __init__(self):
        # ROS Initialization
        rospy.init_node('grounded_sam_service_node')
        # To store the latest image
        self.bridge = CvBridge()
        self.latest_image_msg = ROSImage()
        try:
            self.check_and_get_parameters()
            # Load all models
            self.grounding_dino_model, self.ram_model, self.sam_predictor = self.load_all_models()
            # ROS Publishers Subscribers and Timer
            self.mask_publisher = rospy.Publisher("/grounded_sam/mask", ROSImage, queue_size=10)
            self.image_subscriber = rospy.Subscriber("/my_image_topic", ROSImage, self.image_callback)
            self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        except Exception as e:
            rospy.logerr(f"Error during initialization: {e}")
            rospy.signal_shutdown("Error during initialization")
    
    def get_required_param(self, name, default=None):
        value = rospy.get_param(name, default)
        if value is None:
            rospy.logerr(f"Required parameter {name} is missing!")
            raise ValueError(f"Required parameter {name} is not set on the ROS parameter server!")
        rospy.loginfo(f"Loaded parameter {name}: {value}")
        return value

    def get_float_param(self, name, default=None):
        value = rospy.get_param(name, default)
        if not isinstance(value, (float, int)):
            rospy.logerr(f"Parameter {name} should be a float but got {type(value)}!")
            raise ValueError(f"Parameter {name} is not a float!")
        rospy.loginfo(f"Loaded parameter {name}: {value}")
        return float(value)

    def check_and_get_parameters(self):
        self.config = self.get_required_param("~config")
        self.ram_checkpoint = self.get_required_param("~ram_checkpoint")
        self.grounded_checkpoint = self.get_required_param("~grounded_checkpoint")
        self.sam_checkpoint = self.get_required_param("~sam_checkpoint")
        # self.sam_hq_checkpoint = self.get_required_param("~sam_hq_checkpoint", default=None)
        self.use_sam_hq = self.get_required_param("~use_sam_hq", default=False)
        # self.input_image_path = self.get_required_param("~input_image")
        self.split = self.get_required_param("~split", default=",")
        # self.openai_key = self.get_required_param("~openai_key", default=None)
        # self.openai_proxy = self.get_required_param("~openai_proxy", default=None)
        # self.output_dir = self.get_required_param("~output_dir")
        self.box_threshold = self.get_float_param("~box_threshold", default=0.25)
        self.text_threshold = self.get_float_param("~text_threshold", default=0.2)
        self.iou_threshold = self.get_float_param("~iou_threshold", default=0.5)
        self.device = self.get_required_param("~device", default="cuda")
    
    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.config)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model = model.eval().to(self.device)
        return model

    def load_ram_model(self):
        ram_model = tag2text.ram(pretrained=self.ram_checkpoint, image_size=384, vit='swin_l')
        ram_model = ram_model.eval().to(self.device)
        return ram_model

    def load_sam_model(self):
        if self.use_sam_hq:
            predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        return predictor

    def load_all_models(self):
        grounding_dino_model = self.load_grounding_dino_model()
        ram_model = self.load_ram_model()
        sam_predictor = self.load_sam_model()
        return grounding_dino_model, ram_model, sam_predictor

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)

    def image_callback(self, data):
        self.latest_image_msg = data

    def mask_to_imgmsg(self, mask):
        """
        Convert a mask to a ROS Image message.
        """
        try:
            # Convert the mask to uint8 type
            mask = (mask * 255).astype(np.uint8)
            
            # Check if the mask is grayscale and convert to RGB
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # Convert the mask to an RGB image
            mask_img_msg = self.bridge.cv2_to_imgmsg(mask, encoding="rgb8")
            return mask_img_msg
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return None

    def assign_colors_using_cmap(self, mask_img_np):
        # Normalize the mask values to range [0, 1]
        normalized_mask = mask_img_np.astype(np.float32) / mask_img_np.max()

        # Use matplotlib's colormap to get RGB values
        colored_mask = (cm.viridis(normalized_mask)[:, :, :3] * 255).astype(np.uint8)
        return colored_mask

    def timer_callback(self, event):
        if self.latest_image_msg is None:
            return

        # Process the image using your models
        self.generate_mask_and_tags(self.latest_image_msg)


        # If you have other publishers, you can publish the result here.
    
    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    def load_image_from_msg(self, image_msg):
        # Check if the image message has a valid encoding
        if not image_msg.encoding:
            rospy.logerr("Received image message with empty encoding!")
            return None, None

        # Convert sensor_msgs/Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return None, None
        image_pil = PILImage.fromarray(cv_image)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def generate_mask_and_tags(self, image_msg):
        image_pil, image = self.load_image_from_msg(image_msg)
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

        raw_image = image_pil.resize((384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(self.device)

        # Get tags using RAM model
        res = inference_ram.inference(raw_image, self.ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags = res[0].replace(' |', ',')
        print("Image Tags: ", res[0])

        # Grounding DINO
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            self.grounding_dino_model, image, tags, self.box_threshold, self.text_threshold, device=self.device
        )

        # Initialize SAM
        predictor = self.sam_predictor
        
        try:
            # Convert ROS Image message to OpenCV image (in BGR format)
            image_cv = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            # Convert from BGR to RGB
            image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
        predictor.set_image(image)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        # draw output image
        # plt.figure(figsize=(10, 10))
        # image_np = image
        # plt.imshow(image_np)
        # for mask in masks:
        #     self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     self.show_box(box.numpy(), plt.gca(), label)
        # plt.axis('off')
        # plt.savefig(
        #     os.path.join("/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/outputs/", "rosnewresults.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )
        value = 0  # 0 for background

        mask_img = torch.zeros(masks.shape[-2:])
        for idx, mask in enumerate(masks):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        
        # Convert the mask_img tensor to a NumPy array
        mask_img_np = mask_img.numpy().astype(np.uint8)

        # Print the unique values in mask_img_np
        colored_mask_np = self.assign_colors_using_cmap(mask_img_np)

        # Convert the RGB NumPy array to a ROS Image message
        colored_mask_msg = self.bridge.cv2_to_imgmsg(colored_mask_np, encoding="rgb8")
        self.mask_publisher.publish(colored_mask_msg)
        return

if __name__ == "__main__":
    node = GroundedSAMServiceNode()
    rospy.spin()