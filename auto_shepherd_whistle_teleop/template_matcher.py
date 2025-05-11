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
from vision_msgs.msg import Detection2DArray
from auto_shepherd_msgs.msg import Spectrogram, SpectrogramClassification


class ImageTemplateMatcher(Node):
    def __init__(self):
        super().__init__('image_template_matcher')
        self.bridge = CvBridge()

        # Load config file
        self.config_file = os.getenv('WHISTLE_CONF')
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        t = self.config['template_matching']['topics']
        s = self.config['template_matching']['samples']
        o = self.config['template_matching']['options']

        # Load template samples
        self.templates = self.load_templates(s)

        # Subscriber for incoming images on /pitch_img
        self.subscription = self.create_subscription(Spectrogram, t['input']['raw'], self.image_callback, 10)
        self.last_message_secs = 0

        # Publisher for annotated images on labelled_pitch_image
        self.detected_labels_pub = self.create_publisher(SpectrogramClassification, t['output']['raw'], 10)

        # Publisher for annotated images on labelled_pitch_image
        self.labelled_image_pub_raw = self.create_publisher(Image, t['visual']['raw'], 10)
        self.labelled_image_pub_rgb = self.create_publisher(Image, t['visual']['rgb'], 10)

        # Define extra options
        self.input_delay = o['input_delay'] # delay between inputs (no need to process every image)
        self.draw_style = o['draw_style'] # str to define if every group best is drawn or only the best or all

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
        elif msg.raw_spectrogram.header.stamp.sec - self.input_delay > self.last_message_secs:
            #print('go')
            pass
        else:
            #print('skip')
            return
        self.last_message_secs = msg.raw_spectrogram.header.stamp.sec

        try:
            # Convert the ROS Image message to an OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg.raw_spectrogram, desired_encoding='mono8')
            cv_image_raw = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

            # Format rgb datastream
            viridis = plt.get_cmap('viridis')
            cv_image_rgb = viridis(cv_image)[:, :, :3]
            cv_image_rgb = (cv_image_rgb * 255).astype(np.uint8)
            cv_image_rgb = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)

            # Define storage for bounding boxes of detections
            detection_array = Detection2DArray()
            detection_array.header = msg.raw_spectrogram.header
            best_from_all = {'bv': 0, 'best_scale':None, 'best_val':None,
                             'txt':None, 'text_position':None, 'best_loc':None,
                             'w':None, 'h':None, 'templ':None, 'bounding_box_colour':None}
            print('begin | template scanning')
            # For each template, perform multi-scale matching and draw bounding boxes if matched
            for tmpl in self.templates:
                action = tmpl['action']
                bounding_box_colour = tmpl['bounding_box_colour']
                frequency_range_min = tmpl['frequency_search_range_min']
                frequency_range_max = tmpl['frequency_search_range_max']
                sensitivity = tmpl['sensitivity']
                template = tmpl['image']

                # Identidfy rows that cover range
                nyquist = msg.sample_rate / 2.0
                img_h = cv_image_raw.shape[0]

                def freq_to_row(f_hz):
                    """0 Hz is at the *bottom* of the image; flip if yours is top-origin."""
                    return int(round((1.0 - f_hz / nyquist) * (img_h - 1)))
                row_top    = min(freq_to_row(frequency_range_max), freq_to_row(frequency_range_min))
                row_bottom = max(freq_to_row(frequency_range_max), freq_to_row(frequency_range_min))
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

                        # --- make sure depth & channels match ---------------------------------
                        if roi.ndim == 3:                       # ROI came in RGB – drop colour
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        else:
                            roi_gray = roi

                        if tpl.ndim == 3:                       # template still RGB – drop colour
                            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                        else:
                            tpl_gray = tpl

                        roi_f = roi_gray.astype(np.float32)     # CV_32F depth for both images
                        tpl_f = tpl_gray.astype(np.float32)
                        # ----------------------------------------------------------------------

                        _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(roi_f, tpl_f, cv2.TM_CCOEFF_NORMED))
                        total_attempts += 1

                        if max_val > best_val:
                            best_val, best_loc_roi, best_tpl = max_val, max_loc, tpl
                            best_sx, best_sy = sx, sy

                # translate hit back to full-image coordinates
                best_loc_global = (best_loc_roi[0], best_loc_roi[1] + row_top)


                # If the best match exceeds the sensitivity threshold, draw the bounding box and log the action
                if best_val >= sensitivity:
                    bs = f'{best_scale_x:.2f}x{best_scale_y:.2f}'
                    bv = f'{best_val:.2f})'
                    self.get_logger().info(f'Match at scale {bs} (score: {bv} for action "{action}"')
                    w, h = best_resized_template.shape[::-1]
                    text_position = (best_loc[0], best_loc[1] - 10 if best_loc[1] - 10 > 10 else best_loc[1] + 10)
                    txt = f"{action} {round(best_val,2)}"

                    # Draw detected regions
                    if self.draw_style == 'all':
                        # Draw to mono image
                        cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                        cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                        # Draw to rgb image
                        cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                        cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                        # Add bounding boxes to detection array
                        detection_2d = Detection2D()
                        detection_2d.bbox.centre.position.x = best_loc[0]+int(w/2)
                        detection_2d.bbox.centre.position.y = best_loc[1]+int(h/2)
                        detection_2d.bbox.size_x = w
                        detection_2d.bbox.size_y = h
                        detection_2d.results.hypothesis.class_id = action
                        detection_2d.results.hypothesis.score = bv
                        detection_array.append(detection_2d)

                    # Identify best detection from set to save
                    if best_val > best_from_all['bv']:
                         best_from_all = {'bv':best_val, 'best_scale':bs, 'best_val':bv,
                                          'txt': txt, 'text_position': text_position, 'best_loc': best_loc,
                                          'w':w, 'h':h, 'tmpl':tmpl, 'bounding_box_colour': bounding_box_colour,
                                          'action':action}
                else:
                    self.get_logger().debug(f'No match for {tmpl["file"]} (best score: {best_val:.2f})')

            """ Make this draw the best from each group
            # Draw best detected region
            if self.draw_style == 'best':
                # Draw to mono image
                cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                # Draw to rgb image
                cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                # Add bounding boxes to detection array
                detection_2d = Detection2D()
                detection_2d.bbox.centre.position.x = best_loc[0]+int(w/2)
                detection_2d.bbox.centre.position.y = best_loc[1]+int(h/2)
                detection_2d.bbox.size_x = w
                detection_2d.bbox.size_y = h
                detection_2d.results.hypothesis.class_id = action
                detection_2d.results.hypothesis.score = bv
                detection_array.append(detection_2d)
            """


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

                # Draw best detected region
                if self.draw_style == 'best':
                    # Draw to mono image
                    cv2.rectangle(cv_image_raw, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    cv2.putText(cv_image_raw, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                    # Draw to rgb image
                    cv2.rectangle(cv_image_rgb, best_loc, (best_loc[0] + w, best_loc[1] + h), bounding_box_colour, 2)
                    cv2.putText(cv_image_rgb, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, bounding_box_colour, 1)
                    # Add bounding boxes to detection array
                    detection_2d = Detection2D()
                    detection_2d.bbox.centre.position.x = best_loc[0]+int(w/2)
                    detection_2d.bbox.centre.position.y = best_loc[1]+int(h/2)
                    detection_2d.bbox.size_x = w
                    detection_2d.bbox.size_y = h
                    detection_2d.results.hypothesis.class_id = action
                    detection_2d.results.hypothesis.score = bv
                    detection_array.append(detection_2d)

            # Publish the annotated image to the labelled_pitch_image topic
            annotated_msg_raw = self.bridge.cv2_to_imgmsg(cv_image_raw, encoding='bgr8')
            annotated_msg_raw.header = msg.raw_spectrogram.header
            self.labelled_image_pub_raw.publish(annotated_msg_raw)
            annotated_msg_rgb = self.bridge.cv2_to_imgmsg(cv_image_rgb, encoding='bgr8')
            annotated_msg_rgb.header = msg.raw_spectrogram.header
            self.labelled_image_pub_rgb.publish(annotated_msg_rgb)
            print('published | images')

            # Save image to file (debug)
            save_path = os.path.join(os.path.expanduser("~"), "Desktop", 'whistles', "detected", "bounds_rgb.png")
            cv2.imwrite(save_path, cv_image_rgb)

            # Publish SpectrogramClassification
            spectrogram_classification = SpectrogramClassification()
            spectrogram_classification.spectrogram = msg
            spectrogram_classification.classifications = detection_array
            self.detected_labels_pub.publish(spectrogram_classification)
            print('published | classifications')

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
