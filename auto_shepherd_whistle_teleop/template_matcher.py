import os
import yaml
import cv2
import numpy as np
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class ImageTemplateMatcher(Node):
    def __init__(self):
        super().__init__('image_template_matcher')
        self.bridge = CvBridge()

        # Load config file
        self.config_file = os.getenv('WHISTLE_CONF')
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        topics = self.config['template_matching']['topics']
        samples = self.config['template_matching']['samples']

        # Load template samples
        self.templates = self.load_templates(samples)

        # Subscriber for incoming images on /pitch_img
        self.subscription = self.create_subscription(
            Image,
            topics['input']['raw'],
            self.image_callback,
            10
        )

        # Publisher for annotated images on labelled_pitch_image
        self.labelled_image_pub_raw = self.create_publisher(
            Image,
            topics['output']['raw'],
            10
        )
        self.labelled_image_pub_rgb = self.create_publisher(
            Image,
            topics['output']['rgb'],
            10
        )

        self.get_logger().info('Image Template Matcher Node initialized.')


    def load_templates(self, samples):
        """
        Loads each template image as specified in the configuration.
        Each template entry includes:
          - filename: path to the image (absolute or relative to the package share directory)
          - sensitivity: matching threshold (a value between 0 and 1)
          - bounding_box_colour: BGR colour list for drawing bounding boxes
          - action: an identifier for what to do when the template is matched
        """
        templates = []

        # Determine full path
        if samples['absolute_dir']:
            full_path = samples['absolute_dir']
        else:
            pkg = get_package_share_directory(samples['share_directory_filepath']['package'])
            full_path = os.path.join(pkg, samples['share_directory_filepath']['subpath'])
        self.get_logger().info(f'Loading templates from {full_path}')

        # Identify images
        for group in samples['details']:
            directory = os.path.join(full_path, group['sub_directory'])
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):

                    # Add file to detection
                    template_image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
                    templates.append({
                        'file': filename,
                        'image': template_image,
                        'sensitivity': group['sensitivity'],
                        'bounding_box_colour': group['bounding_box_colour'],
                        'action': group['action']
                    })
                    self.get_logger().info(f'Loaded template {filename} with action "{group["action"]}"')

        return templates

    def image_callback(self, msg):
        """
        Processes images from /pitch_img. The node converts the ROS image to an OpenCV image,
        runs multi-scale template matching for each template, draws bounding boxes for each match
        on a single image, and publishes the annotated image to labelled_pitch_image.
        """
        try:
            # Convert the ROS Image message to an OpenCV BGR image
            cv_image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            # Format rgb datastream
            viridis = plt.get_cmap('viridis')
            cv_image_rgb = viridis(cv_image_raw)[:, :, :3]
            cv_image_rgb = (cv_image_rgb * 255).astype(np.uint8)

            # Convert to grayscale for matching
            # gray_image = cv2.medianBlur(gray_image, 5)

            # For each template, perform multi-scale matching and draw bounding boxes if matched
            for tmpl in self.templates:
                template = tmpl['image']
                sensitivity = tmpl['sensitivity']
                bounding_box_colour = tmpl['bounding_box_colour']
                action = tmpl['action']

                best_val = -np.inf
                best_scale = None
                best_loc = None
                best_resized_template = None

                # Try matching over a range of scales (e.g., 0.5 to 1.5 times the template size)
                for scale in np.linspace(0.5, 1.5, 10):
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    # Skip if the template is larger than the image
                    if resized_template.shape[0] > cv_image_raw.shape[0] or resized_template.shape[1] > cv_image_raw.shape[1]:
                        continue
                    result = cv2.matchTemplate(cv_image_raw, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_val:
                        best_val = max_val
                        best_loc = max_loc
                        best_scale = scale
                        best_resized_template = resized_template

                # If the best match exceeds the sensitivity threshold, draw the bounding box and log the action
                if best_val >= sensitivity:
                    bs = f'{best_scale:.2f}'
                    bv = f'{best_val:.2f})'
                    self.get_logger().info(f'Match at scale {bs} (score: {bv} for action "{action}"')
                    w, h = best_resized_template.shape[::-1]
                    text_position = (best_loc[0], best_loc[1] - 10 if best_loc[1] - 10 > 10 else best_loc[1] + 10)
                    txt = f"{action} {round(best_val,2)}"
                    # Draw to mono image
                    cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                    # Draw to rgb image
                    cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                else:
                    self.get_logger().debug(f'No match for {tmpl["file"]} (best score: {best_val:.2f})')

            # Publish the annotated image to the labelled_pitch_image topic
            annotated_msg_raw = self.bridge.cv2_to_imgmsg(cv_image_raw, encoding='bgr8')
            self.labelled_image_pub_raw.publish(annotated_msg_raw)
            annotated_msg_rgb = self.bridge.cv2_to_imgmsg(cv_image_rgb, encoding='bgr8')
            self.labelled_image_pub_rgb.publish(annotated_msg_rgb)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            raise e

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
