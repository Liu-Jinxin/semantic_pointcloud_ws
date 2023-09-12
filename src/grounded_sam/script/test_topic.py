#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def image_publisher():
    # Initialize the node
    rospy.init_node('image_publisher_node', anonymous=True)
    
    # Create a Publisher, publishing to the /my_image_topic topic, with message type Image
    image_pub = rospy.Publisher('/my_image_topic', Image, queue_size=10)
    
    # Set the rate of publication
    rate = rospy.Rate(10) # 10hz
    
    # Use cv_bridge to convert OpenCV images to ROS messages
    bridge = CvBridge()
    
    # Load the image
    img_path = "/home/appusr/semantic_pointcloud_ws/src/grounded_sam/script/outputs/raw_image.jpg"  # Replace with your image's path
    cv_img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load the image using OpenCV
    
    while not rospy.is_shutdown():
        # Convert the OpenCV image to a ROS Image message
        ros_img = bridge.cv2_to_imgmsg(cv_img, "bgr8")
        rospy.loginfo("This will output to logger rosout.my_logger_name")


        # Publish the image
        image_pub.publish(ros_img)

        # Wait based on the set rate
        rate.sleep()

if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass
