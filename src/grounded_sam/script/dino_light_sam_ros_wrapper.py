#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as ROSImage
from grounded_sam.msg import StringArrayStamped
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from std_msgs.msg import String, Header
from cv_bridge import CvBridge, CvBridgeError
import re
import yaml
import types
import matplotlib.cm as cm

import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
import sys

sys.path.append(
    "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/EfficientSAM"
)
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model


class LightHQSamServiceNode:
    def __init__(self):
        # ROS Initialization
        rospy.init_node("light_hqsam_service_node")
        # To store the latest image
        self.bridge = CvBridge()
        self.latest_image_msg = ROSImage()
        # To store the latest processed timestamp
        self.last_processed_timestamp = None
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        try:
            self.check_and_get_parameters()
            # Load all models
            (
                self.grounding_dino_model,
                self.light_hqsam_predictor,
                self.byte_tracker,
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
        self.text_prompt_str = rospy.get_param("~text_prompt", "[]")
        self.text_prompt = yaml.safe_load(self.text_prompt_str)
        self.mask_callback_timer = self.get_float_param("~mask_callback_timer")
        self.mask_image_topic_name = self.get_required_param("~mask_image_topic_name")
        self.mask_tag_topic_name = self.get_required_param("~mask_tag_topic_name")
        self.raw_image_topic_name = self.get_required_param("~raw_image_topic_name")
        self.dino_config = self.get_required_param("~dino_config")
        self.dino_checkpoint = self.get_required_param("~dino_checkpoint")
        self.sam_checkpoint = self.get_required_param("~sam_checkpoint")
        self.box_threshold = self.get_float_param("~box_threshold", default=0.3)
        self.text_threshold = self.get_float_param("~text_threshold", default=0.25)
        self.nms_threshold = self.get_float_param("~nms_threshold", default=0.8)
        self.track_thresh = self.get_float_param("~track_thresh", default=0.2)
        self.track_buffer = self.get_float_param("~track_buffer", default=30)
        self.match_thresh = self.get_float_param("~match_thresh", default=0.5)
        self.device = self.get_required_param("~device", default="cuda")

    def load_grounding_dino(self):
        grounding_dino_model = Model(
            model_config_path=self.dino_config,
            model_checkpoint_path=self.dino_checkpoint,
        )
        return grounding_dino_model

    def load_light_hqsam(self):
        sam_checkpoint = torch.load(self.sam_checkpoint)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(sam_checkpoint, strict=True)
        light_hqsam.to(device=self.device)
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor

    def load_byte_tracker(self):
        args = types.SimpleNamespace()
        args.track_thresh = self.track_thresh
        args.track_buffer = self.track_buffer
        args.match_thresh = self.match_thresh
        tracker = BYTETracker(args)
        return tracker

    def load_all_models(self):
        grounding_dino_model = self.load_grounding_dino()
        sam_predictor = self.load_light_hqsam()
        byte_tracker = self.load_byte_tracker()
        return grounding_dino_model, sam_predictor, byte_tracker

    def image_callback(self, image_msg):
        self.latest_image_msg = image_msg

    def timer_callback(self, timer_event):
        # Check if there's a valid image message before processing
        if (
            not self.latest_image_msg
            or not self.latest_image_msg.width
            or not self.latest_image_msg.height
        ):
            rospy.loginfo("No valid image received yet.")
            return

        # Check if the image message is newer than the last processed one
        current_timestamp = self.latest_image_msg.header.stamp
        if (
            self.last_processed_timestamp
            and current_timestamp <= self.last_processed_timestamp
        ):
            rospy.loginfo("No new image received.")
            return
        # Process the image using your models
        self.generate_mask_and_tags(self.latest_image_msg)

        # Update the last processed timestamp
        self.last_processed_timestamp = current_timestamp

    def sam_segment(
        self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
    ) -> np.ndarray:
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

    def generate_mask_and_tags(self, image_msg):
        # Convert image_msg to cv2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting image: {e}")
            return
        image_H, image_W = cv_image.shape[:2]
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv_image,
            classes=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.nms_threshold,
            )
            .numpy()
            .tolist()
        )
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # tlwh = np.zeros_like(detections.xyxy)
        # tlwh[:, 0] = detections.xyxy[:, 0]  # top-left x
        # tlwh[:, 1] = detections.xyxy[:, 1]  # top-left y
        # tlwh[:, 2] = detections.xyxy[:, 2] - detections.xyxy[:, 0]  # width
        # tlwh[:, 3] = detections.xyxy[:, 3] - detections.xyxy[:, 1]  # height
        # print("tlwh:", tlwh)
        confidence_reshaped = detections.confidence[:, np.newaxis]
        bounding_box_reshaped = np.copy(detections.xyxy)
        bounding_box_reshaped[:, [0, 2]] /= image_W
        bounding_box_reshaped[:, [1, 3]] /= image_H
        dets = np.hstack((bounding_box_reshaped, confidence_reshaped))
        indices = np.arange(len(dets))
        dets = np.hstack((dets, indices[:, np.newaxis]))
        print("dets:", dets)
        print("image_H:", image_H)
        print("image_W:", image_W)
        online_targets = self.byte_tracker.update(
            dets, [image_H, image_W], [image_H, image_W]
        )
        print("online_targets:", online_targets)
        # print('dir', dir(online_targets))
        # exec ID
        tracker_ids = [target.track_id for target in online_targets]
        print("tracker_ids:", tracker_ids)
        tracker_tlwh = [target._tlwh for target in online_targets]
        print("tracker_tlwh:", tracker_tlwh)
        tracker_indices = [target.index for target in online_targets]
        print("tracker_indices:", tracker_indices)

        # Assume each detection doesn't have a tracker id initially
        detections.tracker_id = [-1] * len(detections.xyxy)

        # Assign the tracker ids according to the indices from online_targets
        for idx, track_id in zip(tracker_indices, tracker_ids):
            detections.tracker_id[int(idx)] = track_id

        print("detections", detections)

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detections.mask = self.sam_segment(
            self.light_hqsam_predictor, cv_image_rgb, detections.xyxy
        )

        # annotate image with detections

        labels = [
            f"{self.text_prompt[class_id]} {confidence:0.2f}"
            + (f" ID:{tracker_id}" if tracker_id != -1 else "")
            for _, confidence, class_id, tracker_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            )
        ]

        # print("labels: ", labels)
        # print("detections: ", detections)
        annotated_image = self.mask_annotator.annotate(
            scene=cv_image.copy(), detections=detections
        )
        annotated_image = self.box_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        # Get the viridis colormap using the recommended method
        # viridis = cm.viridis

        # # Create an empty color image
        # color_mask = np.zeros((cv_image.shape[0], cv_image.shape[1], 3), dtype=np.uint8)

        # # Calculate the number of unique class IDs
        # num_classes = len(np.unique(detections.class_id))

        # # Fill the color image with the masks of each detected object
        # for mask, class_id in zip(detections.mask, detections.class_id):
        #     # Normalize the class ID to get a value between 0 and 1
        #     normalized_value = class_id / float(num_classes)
        #     # Get the RGB color from the viridis colormap and convert to a NumPy array
        #     color = np.array(viridis(normalized_value)[:3]) * 255
        #     color_mask[mask] = color.astype(np.uint8)[::-1]  # Convert RGB to BGR

        # Publish the color mask image
        try:
            self.mask_publisher.publish(
                self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            )
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting image: {e}")
            return

        # # Publish the annotated image
        # try:
        #     self.mask_publisher.publish(
        #         self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        #     )
        # except CvBridgeError as e:
        #     rospy.logerr(f"Error when converting image: {e}")
        #     return


if __name__ == "__main__":
    try:
        node = LightHQSamServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
