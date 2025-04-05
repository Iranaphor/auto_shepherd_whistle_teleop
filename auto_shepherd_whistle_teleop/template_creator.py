import os
import time
import yaml
import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory, get_package_prefix

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image


class ImageTemplateMatcher(Node):
    def __init__(self):
        super().__init__('template_creator')
        self.bridge = CvBridge()

        # Load config file
        self.config_file = os.getenv('WHISTLE_CONF')
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        topics = self.config['template_creator']['topics']
        samples = self.config['template_creator']['samples']

        # Load template samples
        """
        samples:
            absolute_dir: ''
            src_directory_filepath:
                package: 'auto_shepherd_whistle_teleop'
                subpath: 'templates/'

        """
        self.template_save_dir = samples['absolute_dir']
        if not self.template_save_dir:
            pkg = samples['src_directory_filepath']['package']
            ws = get_package_prefix(pkg).split("install")[0].strip()
            cmd = ["colcon", "list", "--packages-select", pkg, "--base-paths", ws, "-p"]
            src_pkg_path = subprocess.check_output(cmd, text=True)
            print(src_pkg_path)
            src = src_pkg_path.strip() #.split(":", 1)[1].strip()
            self.template_save_dir = os.path.join(src, samples['src_directory_filepath']['subpath'])

        # Subscriber for incoming images on /pitch_img
        self.save_frame = False
        self.template_group = None
        self.save_subscription = self.create_subscription(
            String,
            topics['input']['save'],
            self.save_cb,
            10
        )
        self.stream_subscription = self.create_subscription(
            Image,
            topics['input']['raw'],
            self.stream_cb,
            10
        )

        # Publisher for annotated images on labelled_pitch_image
        self.get_logger().info('Image Template Creator Node initialized.')


    def save_cb(self, msg):
        self.template_group = msg.data
        self.save_frame = True

    def stream_cb(self, msg):
        # Check if ready to save frame
        if not self.save_frame:
            return
        self.save_frame = False

        # Make directory if it doesnt exist
        dirpath = os.path.join(self.template_save_dir, self.template_group)
        os.makedirs(dirpath, exist_ok=True)

        # Make unique id for image by looking at how many are in the folder already
        total = len(os.listdir(dirpath))
        raw_name = f'{dirpath}/template_{total}.png'
        self.get_logger().info(f'Saving image to: {raw_name}')

        # Convert the ROS Image message to an OpenCV image to save
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        cv2.imwrite(raw_name, cv_image)


def main(args=None):
    rclpy.init(args=args)
    node = ImageTemplateMatcher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
