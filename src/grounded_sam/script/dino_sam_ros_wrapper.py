#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as ROSImage
from grounded_sam.msg import StringArrayStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge, CvBridgeError

import argparse
import os
import copy
import re

import numpy as np
import json
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import build_sam, build_sam_hq, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DinoSamServiceNode:
    def __init__(self):
        # ROS Initialization
        rospy.init_node("dino_sam_service_node")
        # To store the latest image
        self.bridge = CvBridge()
        self.latest_image_msg = ROSImage()
        try:
            self.check_and_get_parameters()
            # Load all models
            (
                self.grounding_dino_model,
                self.sam_predictor,
            ) = self.load_all_models()
            # ROS Publishers Subscribers and Timer
            self.mask_publisher = rospy.Publisher(
                self.mask_image_topic_name, ROSImage, queue_size=1
            )
            self.mask_tag_publisher = rospy.Publisher(
                self.mask_tag_topic_name, StringArrayStamped, queue_size=1
            )
            self.image_subscriber = rospy.Subscriber(
                self.raw_image_topic_name, ROSImage, self.image_callback
            )
            self.timer = rospy.Timer(
                rospy.Duration(self.mask_callback_timer), self.timer_callback
            )
        except Exception as e:
            rospy.logerr(f"Error during initialization: {e}")
            rospy.signal_shutdown("Error during initialization")

    def get_required_param(self, name, default=None):
        value = rospy.get_param(name, default)
        if value is None:
            rospy.logerr(f"Required parameter {name} is missing!")
            raise ValueError(
                f"Required parameter {name} is not set on the ROS parameter server!"
            )
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
        self.text_prompt = self.get_required_param("~text_prompt")
        self.mask_callback_timer = self.get_float_param("~mask_callback_timer")
        self.mask_image_topic_name = self.get_required_param("~mask_image_topic_name")
        self.mask_tag_topic_name = self.get_required_param("~mask_tag_topic_name")
        self.raw_image_topic_name = self.get_required_param("~raw_image_topic_name")
        self.config = self.get_required_param("~config")
        self.grounded_checkpoint = self.get_required_param("~grounded_checkpoint")
        self.sam_checkpoint = self.get_required_param("~sam_checkpoint")
        self.sam_hq_checkpoint = self.get_required_param("~sam_hq_checkpoint")
        self.use_sam_hq = self.get_required_param("~use_sam_hq", default=False)
        self.split = self.get_required_param("~split", default=",")
        self.box_threshold = self.get_float_param("~box_threshold", default=0.25)
        self.text_threshold = self.get_float_param("~text_threshold", default=0.2)
        self.iou_threshold = self.get_float_param("~iou_threshold", default=0.5)
        self.device = self.get_required_param("~device", default="cuda")
        # self.input_image_path = self.get_required_param("~input_image")
        # self.openai_key = self.get_required_param("~openai_key", default=None)
        # self.openai_proxy = self.get_required_param("~openai_proxy", default=None)
        # self.output_dir = self.get_required_param("~output_dir")

    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.config)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model = model.eval().to(self.device)
        return model

    def load_sam_model(self):
        if self.use_sam_hq:
            predictor = SamPredictor(
                build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device)
            )
        else:
            predictor = SamPredictor(
                build_sam(checkpoint=self.sam_checkpoint).to(self.device)
            )
        return predictor

    def load_all_models(self):
        grounding_dino_model = self.load_grounding_dino_model()
        sam_predictor = self.load_sam_model()
        return grounding_dino_model, sam_predictor

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
        ax.text(x0, y0, label)

    def image_callback(self, data):
        self.latest_image_msg = data

    def mask_to_imgmsg(self, mask):
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
        # Check if there's a valid image message before processing
        if not self.latest_image_msg or not self.latest_image_msg.data:
            rospy.loginfo("No image received yet.")
            return
        # Process the image using your models
        self.generate_mask_and_tags(self.latest_image_msg)

    def get_grounding_output(
        self,
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        with_logits=True,
        device="cuda",
    ):
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
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

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
        # Get grounding output
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.grounding_dino_model,
            image,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device,
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
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
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
        colored_mask_msg.header = image_msg.header
        self.mask_publisher.publish(colored_mask_msg)

        # Extract tags from phrases
        tags = self.extract_tags_from_phrases(pred_phrases)
        tags_msg = StringArrayStamped()
        tags_msg.header = image_msg.header
        tags_msg.size = len(tags)
        tags_msg.Strings = [String(data=tag) for tag in tags]
        self.mask_tag_publisher.publish(tags_msg)
        return

    def extract_tags_from_phrases(self, phrases):
        # extract tags from phrases
        tags = [re.match(r"^(.*?)(?=\()", phrase).group(1).strip() if "(" in phrase else phrase for phrase in phrases]
        return tags


if __name__ == "__main__":
    node = DinoSamServiceNode()
    rospy.spin()
