import os
import time
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        self.input_delay = topics['input']['delay']
        self.last_message_secs = 0

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

        # Load in the default options for each field
        d_bb = samples['default_details']['bounding_box_colour']
        d_fr_min = samples['default_details']['frequency_search_range']['min']
        d_fr_max = samples['default_details']['frequency_search_range']['max']
        d_se = samples['default_details']['sensitivity']

        # Identify images
        for group in samples['details']:

            # Identify properties
            action = group['action']
            bounding_box_colour = group['bounding_box_colour'] if 'bounding_box_colour' in group else d_bb
            frequency_search_range_min = group['frequency_search_range']['min'] if 'frequency_search_range' in group else d_fr_min
            frequency_search_range_max = group['frequency_search_range']['max'] if 'frequency_search_range' in group else d_fr_max
            sensitivity = group['sensitivity'] if 'sensitivity' in group else d_se
            sub_directory = group['sub_directory'] if 'sub_directory' in group else f'{action}/'

            # Loop through directory and save details
            directory = os.path.join(full_path, sub_directory)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):

                    # Add file to detection
                    template_image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
                    templates.append({
                        'action': action,
                        'bounding_box_colour': bounding_box_colour,
                        'frequency_search_range_min': frequency_search_range_min,
                        'frequency_search_range_max': frequency_search_range_max,
                        'sensitivity': sensitivity,
                        'sub_directory': sub_directory,
                        'file': filename,
                        'image': template_image
                    })
                    self.get_logger().info(f'Loaded template {filename} with action "{group["action"]}"')
        return templates

    def image_callback(self, msg):
        """
        Processes images from /pitch_img. The node converts the ROS image to an OpenCV image,
        runs multi-scale template matching for each template, draws bounding boxes for each match
        on a single image, and publishes the annotated image to labelled_pitch_image.
        """
        start_time = time.time()
        total_attempts = 0

        #print(self.last_message_secs)
        #print(self.input_delay)
        #print(msg.header.stamp.sec)

        if self.last_message_secs == 0:
            #print('first')
            pass
        elif msg.header.stamp.sec - self.input_delay > self.last_message_secs:
            #print('go')
            pass
        else:
            #print('skip')
            return
        self.last_message_secs = msg.header.stamp.sec

        try:
            # Convert the ROS Image message to an OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_image_raw = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

            # Format rgb datastream
            viridis = plt.get_cmap('viridis')
            cv_image_rgb = viridis(cv_image)[:, :, :3]
            cv_image_rgb = (cv_image_rgb * 255).astype(np.uint8)
            cv_image_rgb = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)

            # Convert to grayscale for matching
            # gray_image = cv2.medianBlur(gray_image, 5)

            best_from_all = {'bv': 0, 'best_scale':None, 'best_val':None,
                             'txt':None, 'text_position':None, 'best_loc':None,
                             'w':None, 'h':None, 'templ':None, 'bounding_box_colour':None}

            # For each template, perform multi-scale matching and draw bounding boxes if matched
            for tmpl in self.templates:
                action = tmpl['action']
                bounding_box_colour = tmpl['bounding_box_colour']
                frequency_range_min = tmpl['frequency_search_range_min']
                frequency_range_max = tmpl['frequency_search_range_max']
                sensitivity = tmpl['sensitivity']
                template = tmpl['image']

                # Identidfy rows that cover range
                nyquist = self.sample_rate / 2.0
                img_h = cv_image_raw.shape[0]

                def freq_to_row(f_hz):
                    """0 Hz is at the *bottom* of the image; flip if yours is top-origin."""
                    return int(round((1.0 - f_hz / nyquist) * (img_h - 1)))
                row_top    = min(freq_to_row(freq_max), freq_to_row(freq_min))
                row_bottom = max(freq_to_row(freq_max), freq_to_row(freq_min))
                roi = cv_image_raw[row_top:row_bottom + 1, :]

                # Loop through each size to try for detection
                best_val = -np.inf
                best_scale = None
                best_loc = None
                best_resized_template = None
                for sx in np.linspace(0.15, 1.0, 10):
                    for sy in np.linspace(0.15, 1.0, 10):

                        tpl = cv2.resize(template, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
                        if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                            continue

                        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED))
                        total_attempts += 1

                        if max_val > best_val:
                            best_val, best_loc_roi, best_tpl = max_val, max_loc, tpl
                            best_sx, best_sy = sx, sy

                # translate hit back to full-image coordinates
                best_loc_global = (best_loc_roi[0], best_loc_roi[1] + row_top)



                """
                # Try matching over a range of scales (e.g., 0.5 to 1.5 times the template size)
                for scale_x in np.linspace(0.15, 1.0, 10):
                    for scale_y in np.linspace(0.15, 1.0, 10):
                        resized_template = cv2.resize(template, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
                        # Skip if the template is larger than the image
                        if resized_template.shape[0] > cv_image_raw.shape[0] or resized_template.shape[1] > cv_image_raw.shape[1]:
                            continue
                        result = cv2.matchTemplate(cv_image, resized_template, cv2.TM_CCOEFF_NORMED)
                        total_attempts += 1
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        if max_val > best_val:
                            best_val = max_val
                            best_loc = max_loc
                            best_scale_x = scale_x
                            best_scale_y = scale_y
                            best_resized_template = resized_template
                """

                # If the best match exceeds the sensitivity threshold, draw the bounding box and log the action
                if best_val >= sensitivity:
                    bs = f'{best_scale_x:.2f}x{best_scale_y:.2f}'
                    bv = f'{best_val:.2f})'
                    self.get_logger().info(f'Match at scale {bs} (score: {bv} for action "{action}"')
                    w, h = best_resized_template.shape[::-1]
                    text_position = (best_loc[0], best_loc[1] - 10 if best_loc[1] - 10 > 10 else best_loc[1] + 10)
                    txt = f"{action} {round(best_val,2)}"
                    # Draw to mono image
                    #cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    #cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                    # Draw to rgb image
                    #cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    #cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                    if best_val > best_from_all['bv']:
                         best_from_all = {'bv':best_val, 'best_scale':bs, 'best_val':bv, 'txt': txt, 'text_position': text_position, 'best_loc': best_loc, 'w':w, 'h':h, 'tmpl':tmpl, 'bounding_box_colour': bounding_box_colour, 'action':action}
                else:
                    self.get_logger().debug(f'No match for {tmpl["file"]} (best score: {best_val:.2f})')


            # If the best match exceeds the sensitivity threshold, draw the bounding box and log the action
            if best_from_all['bv'] > 0:
                bs = best_from_all['best_scale']
                bv = best_from_all['best_val']
                action = best_from_all['action']
                self.get_logger().info(f'Best Match at scale {bs} (score: {bv} for action "{action}"')
                w, h = best_from_all['w'], best_from_all['h']
                best_loc = best_from_all['best_loc']
                text_position = best_from_all['text_position']
                txt = best_from_all['txt']
                bounding_box_colour = best_from_all['bounding_box_colour']
                # Draw to mono image
                cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                # Draw to rgb image
                cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)

            # Publish the annotated image to the labelled_pitch_image topic
            annotated_msg_raw = self.bridge.cv2_to_imgmsg(cv_image_raw, encoding='bgr8')
            self.labelled_image_pub_raw.publish(annotated_msg_raw)
            annotated_msg_rgb = self.bridge.cv2_to_imgmsg(cv_image_rgb, encoding='bgr8')
            self.labelled_image_pub_rgb.publish(annotated_msg_rgb)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            raise e
        print(f"--- {(time.time() - start_time):.4f}s for {total_attempts} total attempts ---")

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
