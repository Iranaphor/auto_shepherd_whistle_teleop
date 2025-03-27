import os
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

class ImageTemplateMatcher(Node):
    def __init__(self, config_file, package_name='auto_shepherd_whistle_teleop'):
        super().__init__('image_template_matcher')
        self.bridge = CvBridge()
        self.package_name = package_name
        self.config = self.load_config(config_file)
        self.template_infos = self.config.get('templates', [])
        self.templates = self.load_templates(self.template_infos)
        
        # Subscriber for incoming images on /pitch_img
        self.subscription = self.create_subscription(
            Image,
            '/pitch_image',
            self.image_callback,
            10
        )
        # Publisher for annotated images on labelled_pitch_image
        self.image_pub = self.create_publisher(Image, 'labelled_pitch_image', 10)
        
        self.get_logger().info('Image Template Matcher Node initialized.')

    def load_config(self, config_file):
        """
        Loads the YAML configuration.
        If the path is relative, it will be taken from the package share directory.
        """
        if not os.path.isabs(config_file):
            config_file = os.path.join(get_package_share_directory(self.package_name), config_file)
        self.get_logger().info(f'Loading configuration from {config_file}')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_templates(self, template_infos):
        """
        Loads each template image as specified in the configuration.
        Each template entry includes:
          - filename: path to the image (absolute or relative to the package share directory)
          - sensitivity: matching threshold (a value between 0 and 1)
          - bounding_box_colour: BGR colour list for drawing bounding boxes
          - action: an identifier for what to do when the template is matched
        """
        templates = []
        for info in template_infos:
            filename = info.get('filename')
            sensitivity = info.get('sensitivity', 0.8)  # default threshold if not provided
            bounding_box_colour = info.get('bounding_box_colour', [0, 255, 0])
            action = info.get('action', None)
            
            # Determine full path: if filename starts with '/', it's an absolute path.
            if os.path.isabs(filename):
                full_path = filename
            else:
                full_path = os.path.join(get_package_share_directory(self.package_name), filename)
            
            # Load the template in grayscale
            template_image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if template_image is not None:
                templates.append({
                    'file': full_path,
                    'image': template_image,
                    'sensitivity': sensitivity,
                    'bounding_box_colour': bounding_box_colour,
                    'action': action
                })
                self.get_logger().info(f'Loaded template {filename} with action "{action}"')
            else:
                self.get_logger().warn(f'Failed to load template {full_path}')
        return templates

    def image_callback(self, msg):
        """
        Processes images from /pitch_img. The node converts the ROS image to an OpenCV image,
        runs multi-scale template matching for each template, draws bounding boxes for each match
        on a single image, and publishes the annotated image to labelled_pitch_image.
        """
        print('img callback')
        try:
            # Convert the ROS Image message to an OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert to grayscale for matching
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
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
                    if resized_template.shape[0] > gray_image.shape[0] or resized_template.shape[1] > gray_image.shape[1]:
                        continue
                    result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_val:
                        best_val = max_val
                        best_loc = max_loc
                        best_scale = scale
                        best_resized_template = resized_template

                # If the best match exceeds the sensitivity threshold, draw the bounding box and log the action
                if best_val >= sensitivity:
                    self.get_logger().info(
                        f'Match for {tmpl["file"]} with action "{action}" at scale {best_scale:.2f} (score: {best_val:.2f})'
                    )
                    w, h = best_resized_template.shape[::-1]
                    cv2.rectangle(cv_image, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    text_position = (best_loc[0], best_loc[1] - 10 if best_loc[1] - 10 > 10 else best_loc[1] + 10)
                    cv2.putText(cv_image, action, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bounding_box_colour, 2, cv2.LINE_AA)
                else:
                    self.get_logger().debug(f'No match for {tmpl["file"]} (best score: {best_val:.2f})')
            
            # Publish the annotated image to the labelled_pitch_image topic
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.image_pub.publish(annotated_msg)
            
            # (Optional) Display the annotated image locally for debugging
            #cv2.imshow("Matched", cv_image)
            #cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    # Specify the config file relative to the package share directory,
    # or provide an absolute path if desired.
    config_file = 'config/template_matching_config.yaml'  # update to your config file location
    node = ImageTemplateMatcher(config_file)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
