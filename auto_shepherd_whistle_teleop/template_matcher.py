import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import glob
from ament_index_python.packages import get_package_share_directory


class ImageTemplateMatcher(Node):
    def __init__(self, template_dir):
        super().__init__('image_template_matcher')
        self.template_dir = template_dir
        self.bridge = CvBridge()
        self.template_images = self.load_templates(template_dir)
        
        # Subscribe to the /pitch_img topic
        self.subscription = self.create_subscription(
            Image,
            '/pitch_img',
            self.image_callback,
            10  # QoS depth
        )
        self.get_logger().info('Image Template Matcher Node initialized.')

    def load_templates(self, template_dir):
        """
        Loads all template images from the specified directory.
        The method searches for .png, .jpg, and .jpeg files.
        """
        templates = []
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in image_extensions:
            for file in glob.glob(os.path.join(template_dir, ext)):
                # Read image in grayscale for template matching
                template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates.append((file, template))
                    self.get_logger().info(f'Loaded template: {file}')
                else:
                    self.get_logger().warn(f'Failed to load template: {file}')
        return templates

    def image_callback(self, msg):
        """
        Callback function for processing images from /pitch_img.
        It converts the image message to an OpenCV image, converts it to grayscale,
        and then performs template matching against each loaded template.
        """
        try:
            # Convert the ROS Image message to an OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Loop over each template image
            for file, template in self.template_images:
                # Get the dimensions of the template
                w, h = template.shape[::-1]
                # Perform template matching using normalized cross-correlation
                result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  # Adjust threshold as needed
                loc = np.where(result >= threshold)
                
                if len(loc[0]) > 0:
                    self.get_logger().info(f'Template match found for {file}')
                    # (Optional) For visualization, you could draw rectangles on the matches:
                    # for pt in zip(*loc[::-1]):
                    #     cv2.rectangle(cv_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                    # cv2.imshow("Matched", cv_image)
                    # cv2.waitKey(1)
                else:
                    self.get_logger().debug(f'No match found for {file}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    # Specify the directory containing the template images.
    # Change '/path/to/template_directory' to your actual path.
    template_dir = os.path.join(
        get_package_share_directory('auto_shepherd_whistle_teleop'),
        'templates'
    )
    node = ImageTemplateMatcher(template_dir)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
