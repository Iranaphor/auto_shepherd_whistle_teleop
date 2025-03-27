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
            'pitch_image',
            self.image_callback,
            10  # QoS depth
        )
        self.publisher = self.create_subscription(
            Image,
            'labelled_pitch_image',
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
        Processes images from /pitch_img, converting the image to grayscale,
        and performing multi-scale template matching against each loaded template.
        """
        print('image callback')
        try:
            # Convert the ROS Image message to an OpenCV BGR image then to greyscale
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Loop over each template
            for file, template in self.template_images:
                best_val = -np.inf
                best_scale = None
                best_loc = None
                best_resized_template = None

                # Try a range of scales (for example, from 0.5 to 1.5 times the original size)
                for scale in np.linspace(0.5, 1.5, 10):
                    # Resize the template according to the current scale
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    # Skip if the resized template is larger than the image
                    if resized_template.shape[0] > gray_image.shape[0] or resized_template.shape[1] > gray_image.shape[1]:
                        continue
                    #print(file.split('/')[-1], round(scale,2))

                    # Perform template matching
                    result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    # Check if this scale gives a better match
                    if max_val > best_val:
                        best_val = max_val
                        best_loc = max_loc
                        best_scale = scale
                        best_resized_template = resized_template

                threshold = 0.1  # Adjust threshold as needed
                filename = file.split('/')[-1]
                if best_val >= threshold:
                    self.get_logger().info(f'Match found for {filename} at scale {best_scale:.2f} with score {best_val:.2f}')
                    # (Optional) Visualization: draw rectangle on the match location
                    w, h = best_resized_template.shape[::-1]
                    cv2.rectangle(cv_image, best_loc, (best_loc[0] + w, best_loc[1] + h), (0, 255, 0), 2)
                    cv2.imshow("Matched", cv_image)
                    cv2.waitKey(1)
                else:
                    self.get_logger().debug(f'No match found for {filename} (best score: {best_val:.2f})')

            # Convert to ROS2 Image and publish
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
            self.image_pub.publish(ros_image)

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
