#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from grounded_sam.msg import ObjectInfo, ObjectsArrayStamped
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from std_msgs.msg import String, Header
from cv_bridge import CvBridge, CvBridgeError
import yaml
import types

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
        self.latest_depth_image_msg = ROSImage()
        self.obj_array_msg = ObjectsArrayStamped()
        self.last_processed_timestamp = None  # To store the latest processed timestamp
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        self.camera_info_callback_count = 0
        self.MAX_CAMERA_INFO_CALLBACKS = 3  # Change the warmup count here
        self.pc_without_scale = None
        try:
            self.check_and_get_parameters()
            # Load all models
            (
                self.grounding_dino_model,
                self.light_hqsam_predictor,
                self.byte_tracker,
            ) = self.load_all_models()
            # ROS Publishers Subscribers and Timer
            self.mask_color_publisher = rospy.Publisher(
                self.mask_color_image_topic_name, ROSImage, queue_size=1
            )
            self.mask_depth_publisher = rospy.Publisher(
                self.mask_depth_image_topic_name, ROSImage, queue_size=1
            )
            self.objects_info_publisher = rospy.Publisher(
                self.objects_info_topic_name, ObjectsArrayStamped, queue_size=1
            )
            self.camera_info_subscriber = rospy.Subscriber(
                self.raw_image_info_topic_name, CameraInfo, self.camera_info_callback
            )
            self.image_subscriber = message_filters.Subscriber(
                self.raw_image_topic_name, ROSImage
            )
            self.depth_image_subscriber = message_filters.Subscriber(
                self.depth_image_topic_name, ROSImage
            )
            self.ts = message_filters.TimeSynchronizer(
                [self.image_subscriber, self.depth_image_subscriber], 10
            )
            self.ts.registerCallback(self.time_synchronizer_callback)
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
        self.mask_color_image_topic_name = self.get_required_param(
            "~mask_color_image_topic_name"
        )
        self.mask_depth_image_topic_name = self.get_required_param(
            "~mask_depth_image_topic_name"
        )
        self.objects_info_topic_name = self.get_required_param("~objects_info_topic_name")
        self.raw_image_topic_name = self.get_required_param("~raw_image_topic_name")
        self.raw_image_info_topic_name = self.get_required_param(
            "~raw_image_info_topic_name"
        )
        self.depth_image_topic_name = self.get_required_param("~depth_image_topic_name")
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

    def camera_info_callback(self, camera_info_msg):
        try:
            # Check if we reached the max callbacks
            if self.camera_info_callback_count >= self.MAX_CAMERA_INFO_CALLBACKS:
                self.camera_info_subscriber.unregister()
                return
            camera_info_k = np.array(camera_info_msg.K).reshape(3, 3)

            # Convert the inverse matrix to a PyTorch tensor and send to CUDA
            inv_camera_info_k = torch.tensor(
                np.linalg.inv(camera_info_k), dtype=torch.float32
            ).cuda()
            index_h = torch.arange(0, camera_info_msg.height)
            index_w = torch.arange(0, camera_info_msg.width)
            yy, xx = torch.meshgrid(index_h, index_w)
            coordinates = torch.stack((xx, yy), dim=0).permute(1, 2, 0).cuda()
            hom_coord = (
                torch.cat((coordinates, torch.ones_like(coordinates[..., 0:1])), dim=-1)
                .permute(0, 2, 1)
                .float()
            )
            self.pc_without_scale = torch.matmul(
                inv_camera_info_k.unsqueeze(0), hom_coord
            ).permute(0, 2, 1)

            # Increment the count
            self.camera_info_callback_count += 1

        except Exception as e:
            rospy.logerr(f"Error when loading camera info: {e}")

    def time_synchronizer_callback(self, image_msg, depth_image_msg):
        self.latest_image_msg = image_msg
        self.latest_depth_image_msg = depth_image_msg

    def timer_callback(self, timer_event):
        # Check if there's a valid image message before processing
        if (
            not self.latest_image_msg
            or not self.latest_image_msg.width
            or not self.latest_image_msg.height
        ):
            rospy.loginfo("No valid color image received yet.")
            return

        if (
            not self.latest_depth_image_msg
            or not self.latest_depth_image_msg.width
            or not self.latest_depth_image_msg.height
        ):
            rospy.loginfo("No valid depth image received yet.")
            return

        # Check if the image message is newer than the last processed one
        current_timestamp = self.latest_image_msg.header.stamp
        if (
            self.last_processed_timestamp
            and current_timestamp == self.last_processed_timestamp
        ):
            rospy.loginfo("No new image received.")
            return
        # Process the image using your models
        self.generate_mask_and_tags(self.latest_image_msg, self.latest_depth_image_msg)

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

    def convert_image_msg_to_cv(self, image_msg):
        try:
            return self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting image: {e}")
            return None

    def convert_depth_image_msg_to_cv(self, depth_image_msg):
        try:
            # First, convert the ROS Image message to a CV image
            depth_image_single_channel = self.bridge.imgmsg_to_cv2(
                depth_image_msg, desired_encoding="passthrough"
            )
            depth_tensor = torch.from_numpy(
                depth_image_single_channel.astype(np.float32)
            ).cuda()
            # Normalize the depth image to 0-255 range
            normalized_depth_image = cv2.normalize(
                depth_image_single_channel, None, 0, 255, cv2.NORM_MINMAX
            )
            # Convert the normalized depth image to 8-bit
            normalized_depth_image_8bit = normalized_depth_image.astype(np.uint8)
            # Now convert the single channel depth image to a 3 channel grayscale image
            depth_image_three_channel = cv2.cvtColor(
                normalized_depth_image_8bit, cv2.COLOR_GRAY2BGR
            )
            return depth_image_three_channel, depth_tensor
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting depth image: {e}")
            return None

    def get_detections(self, cv_image):
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv_image,
            classes=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

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
        return detections

    def reshape_detections(self, detections, image_H, image_W):
        confidence_reshaped = detections.confidence[:, np.newaxis]
        bounding_box_reshaped = np.copy(detections.xyxy)
        bounding_box_reshaped[:, [0, 2]] /= image_W
        bounding_box_reshaped[:, [1, 3]] /= image_H
        dets = np.hstack((bounding_box_reshaped, confidence_reshaped))
        indices = np.arange(len(dets))
        return np.hstack((dets, indices[:, np.newaxis]))

    def assign_tracker_ids(self, detections, online_targets):
        # Assume each detection doesn't have a tracker id initially
        detections.tracker_id = [-1] * len(detections.xyxy)

        # Assign the tracker ids according to the indices from online_targets
        tracker_ids = [target.track_id for target in online_targets]
        tracker_indices = [target.index for target in online_targets]

        for idx, track_id in zip(tracker_indices, tracker_ids):
            detections.tracker_id[int(idx)] = track_id

    def annotate_image(self, cv_image, depth_image, detections):
        # labels = [
        #     f"{self.text_prompt[class_id]} {confidence:0.2f}"
        #     + (f" ID:{tracker_id}" if tracker_id != -1 else "")
        #     for _, confidence, class_id, tracker_id in zip(
        #         detections.xyxy,
        #         detections.confidence,
        #         detections.class_id,
        #         detections.tracker_id,
        #     )
        # ]
        labels = [
            f"{self.text_prompt[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            )
        ]
        annotated_color_image = self.mask_annotator.annotate(
            cv_image.copy(), detections
        )
        annotated_color_image = self.box_annotator.annotate(
            annotated_color_image, detections, labels
        )
        annotated_depth_image = self.mask_annotator.annotate(
            depth_image.copy(), detections
        )
        return annotated_color_image, annotated_depth_image

    def publish_ros_msg(
        self, annotated_color_image, annotated_depth_image, image_msg, depth_image_msg, objects_info_msg
    ):
        try:
            color_img_msg = self.bridge.cv2_to_imgmsg(annotated_color_image, "bgr8")
            color_img_msg.header = (
                image_msg.header
            )  # Copy the header from the original image message
            self.mask_color_publisher.publish(color_img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting color image: {e}")
        try:
            depth_img_msg = self.bridge.cv2_to_imgmsg(annotated_depth_image, "bgr8")
            depth_img_msg.header = (
                depth_image_msg.header
            )  # Copy the header from the original depth image message
            self.mask_depth_publisher.publish(depth_img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error when converting depth image: {e}")
        try:
            self.obj_array_msg.objects = objects_info_msg
            self.obj_array_msg.header = image_msg.header
            self.obj_array_msg.size = len(objects_info_msg)
            self.objects_info_publisher.publish(self.obj_array_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error when publish objects info: {e}")

    def generate_mask_and_tags(self, image_msg, depth_image_msg):
        cv_image = self.convert_image_msg_to_cv(image_msg)
        depth_image, depth_tensor = self.convert_depth_image_msg_to_cv(depth_image_msg)

        image_H, image_W = cv_image.shape[:2]
        detections = self.get_detections(cv_image)

        dets = self.reshape_detections(detections, image_H, image_W)
        online_targets = self.byte_tracker.update(
            dets, [image_H, image_W], [image_H, image_W]
        )

        self.assign_tracker_ids(detections, online_targets)

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detections.mask = self.sam_segment(
            self.light_hqsam_predictor, cv_image_rgb, detections.xyxy
        )
        point_cloud_3d = self.generate_3dpoint_for_image(depth_tensor)

        objects_info_msg = []
        
        print("detections: ", detections)
        print("detections.xyxy: ", detections.xyxy)
        print("detections.mask: ", detections.mask)
        for i in range(len(detections.class_id)):
            object_info_msg = ObjectInfo()
            object_info_msg.semantic_label = self.text_prompt[detections.class_id[i]]  # Assuming class_id is an index into text_prompt
            object_info_msg.semantic_id = detections.class_id[i]
            object_info_msg.track_id = detections.tracker_id[i]
            mask = detections.mask[i]
            object_points_3d = point_cloud_3d[mask]
            print("object_points_3d: ", object_points_3d)

            objects_info_msg.append(object_info_msg)
        

        annotated_color_image, annotated_depth_image = self.annotate_image(
            cv_image, depth_image, detections
        )
        self.publish_ros_msg(
            annotated_color_image, annotated_depth_image, image_msg, depth_image_msg, objects_info_msg
        )

    def generate_3dpoint_for_image(self, depth_tensor):
        if self.pc_without_scale is None:
            return None
        else:
            depth_expanded = depth_tensor.unsqueeze(-1)
            point_3d = depth_expanded * self.pc_without_scale
        return point_3d


if __name__ == "__main__":
    try:
        node = LightHQSamServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
