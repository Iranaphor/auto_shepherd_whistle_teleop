#!/usr/bin/env python3
#
# real_time_spectrogram.py  –  subscribe to AudioChunk and publish spectrogram
#
#   subscribes:  /input/audio   (auto_shepherd_msgs/AudioChunk)
#   publishes :  see topic block in __init__
#

import os
import yaml
import cv2
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Import message types
from auto_shepherd_msgs.msg import PitchTrack, SpectralPeak, AudioChunk, Spectrogram
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time as RosTime


class RealTimeSpectrogram(Node):
    def __init__(self):
        super().__init__('real_time_spectrogram_node')
        self.get_logger().info('Real Time Spectrogram Node initialized.')

        # --------------------------------------------------
        # 1.  Load configuration
        # --------------------------------------------------
        self.config_file = os.getenv('WHISTLE_CONF')
        if not self.config_file or not os.path.isfile(self.config_file):
            raise RuntimeError('Environment variable WHISTLE_CONF must point to a YAML file')
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

        p = self.config['audio_preprocessor']['topics']
        c = self.config['codex_detector']['topics']

        # --------------------------------------------------
        # 2.  Publishers & subscriber
        # --------------------------------------------------
        # Subscriber for incoming audio from input node (raw audio chunk)
        self.create_subscription(AudioChunk, p['input']['audio'], self.audio_input_cb, 10)

        # Publisher for spectrogram (raw)
        self.spectrogram_pub = self.create_publisher(Spectrogram, p['output']['raw'], self.get_image_qos())

        # Publishers for stream (raw/rgb)
        self.image_stream_raw = self.create_publisher(Image, p['visual']['st_raw'], self.get_image_qos())
        self.image_stream_rgb = self.create_publisher(Image, p['visual']['st_rgb'], self.get_image_qos())

        # Publishers for preprocessed stream (raw/rgb)
        self.image_preprocessed_raw = self.create_publisher(Image, p['visual']['pr_raw'], self.get_image_qos())
        self.image_preprocessed_rgb = self.create_publisher(Image, p['visual']['pr_rgb'], self.get_image_qos())

        # Publisher for detected activity (codex, raw/rgb)
        self.pitch_pub = self.create_publisher(PitchTrack, c['output']['codex'], self.get_qos())
        self.image_codex_raw = self.create_publisher(Image, c['visual']['raw'], self.get_image_qos())
        self.image_codex_rgb = self.create_publisher(Image, c['visual']['rgb'], self.get_image_qos())


        # --------------------------------------------------
        # 3.  Spectrogram processing parameters
        # --------------------------------------------------
        self.bridge = CvBridge()

        # Audio and display parameters
        p_conf = self.config['audio_preprocessor']['preprocessing']
        c_conf = self.config['codex_detector']['activity_detection']

        # Declare colour format for rgb
        self.viridis = plt.get_cmap('viridis')

        # Setup preprocessing details
        self.threshold_db = p_conf['threshold_db']
        self.secondary_threshold_db = p_conf['secondary_threshold_db']

        # Save preprocessing properties
        self.medfilt_do = p_conf['medfilt']['do']
        self.medfilt_kernel = p_conf['medfilt']['kernel']

        self.normalize_do = p_conf['normalize']['do']
        self.normalize_min = p_conf['normalize']['min']
        self.normalize_max = p_conf['normalize']['max']

        self.frequency_crop_do = p_conf['frequency_crop']['do']
        self.frequency_crop_min = p_conf['frequency_crop']['min']
        self.frequency_crop_max = p_conf['frequency_crop']['max']

        # Save activity detection properties
        self.run_activity_detection = c_conf['run']
        self.record_duration = c_conf['record_duration']

        # These will be set on first incoming chunk
        self.initialised = False
        self.sample_rate = None
        self.chunk_size = None
        self.window_duration = None
        self.num_chunks = None
        self.freq_mask = None
        self.trimmed_freqs = None
        self.spectrogram = None
        self.frequency_min = None
        self.frequency_max = None

        # Event detection variables
        self.recording = False
        self.record_start_time = None

        # --------------------------------------------------
        # 4.  Timer to call update() every 50 ms
        # --------------------------------------------------
        self.create_timer(0.05, self.update)

    # ======================================================
    #                   QoS helper methods
    # ======================================================
    def get_qos(self):
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

    def get_image_qos(self):
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

    # ======================================================
    #                 CALLBACKS / HELPERS
    # ======================================================
    def audio_input_cb(self, msg):
        #print(len(msg.data), msg.frames, msg.channels, msg.sample_rate)
        if not msg.data:          # ignore “flush-empty” message
            return

        # Initialise buffers on first packet
        if not self.initialised:
            self._init_from_msg(msg)

        # Called as callback from audio_input.py / audio_input_mic.py
        indata = np.asarray(msg.data, dtype=np.float32)\
                    .reshape((msg.frames, msg.channels))
        frames = msg.frames
        # fabricate minimal time_info / status as before
        time_info = {'input_buffer_adc_time': msg.header.stamp.sec +
                                             msg.header.stamp.nanosec * 1e-9}
        status = "from_file"

        self.audio_callback(indata, frames, time_info, status)

    def _init_from_msg(self, msg):
        # Audio and display parameters
        s_conf = self.config['audio_preprocessor']['spectrogram']

        # Save input stream properties
        self.sample_rate = msg.sample_rate
        self.chunk_size = msg.frames
        self.window_duration = s_conf['window_duration']
        self.num_chunks = int(self.window_duration *
                              self.sample_rate / self.chunk_size)

        # Compute full FFT frequency bins and create frequency mask
        full_freqs = np.fft.rfftfreq(self.chunk_size, d=1/self.sample_rate)
        if self.frequency_crop_do:
            self.freq_mask = (full_freqs >= self.frequency_crop_min) \
                             & (full_freqs <= self.frequency_crop_max)
            self.trimmed_freqs = full_freqs[self.freq_mask]
            self.frequency_min = self.frequency_crop_min
            self.frequency_max = self.frequency_crop_max
        else:
            self.trimmed_freqs = full_freqs
            self.frequency_min = self.trimmed_freqs[0]
            self.frequency_max = self.trimmed_freqs[-1]

        # Initialize spectrogram buffer (rows: time slices, columns: frequency bins)
        self.spectrogram = np.zeros((self.num_chunks, len(self.trimmed_freqs)))

        self.initialised = True
        self.get_logger().info(f'Initialised spectrogram: '
                               f'rate={self.sample_rate} Hz, '
                               f'chunk={self.chunk_size} frames, '
                               f'window={self.window_duration} s')

    # ──────────────────────────────────────────────────────────────────────────
    # Core: process incoming chunk
    # ──────────────────────────────────────────────────────────────────────────
    def audio_callback(self, indata, frames, time_info, status):
        # Process mono audio
        audio_data = indata[:, 0]
        windowed = audio_data * np.hanning(len(audio_data))
        fft_data = np.abs(np.fft.rfft(windowed))
        fft_data = 20 * np.log10(fft_data + 1e-6)
        fft_data = np.where(fft_data < self.threshold_db, 0, fft_data)
        if self.frequency_crop_do:
            fft_data_trimmed = fft_data[self.freq_mask]
        else:
            fft_data_trimmed = fft_data
        # Roll the spectrogram buffer and append the new data
        self.spectrogram[:-1, :] = self.spectrogram[1:, :]
        self.spectrogram[-1, :] = fft_data_trimmed


    def publish_raw_and_rgb(self, img, raw_pub, rgb_pub, header):
        # Apply colour and format as uint8
        colored = self.viridis(img)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        # Convert to Image and publish Raw
        ros_image = self.bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="mono8")
        ros_image.header = header
        raw_pub.publish(ros_image)

        # Convert to Image and publish RGB
        ros_image = self.bridge.cv2_to_imgmsg(colored, encoding="rgb8")
        ros_image.header = header
        rgb_pub.publish(ros_image)

    def publish_spectrogram(self, img, header, sample_rate, chunk_size):
        S = Spectrogram()
        # Load Image
        S.raw_spectrogram = self.bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="mono8")
        S.raw_spectrogram.header = header
        # Load properties
        S.window_duration = float(self.window_duration) #from config file
        S.sample_rate     = sample_rate          #from audio source
        S.chunk_size      = chunk_size           #from audio source
        # Publish
        self.spectrogram_pub.publish(S)

    # ──────────────────────────────────────────────────────────────────────────
    # Timer-driven update:  visualisation + event detection
    # ──────────────────────────────────────────────────────────────────────────
    def update(self):
        if not self.initialised:
            return

        # Format datastream
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "spectrogram"

        # Publish input stream
        self.publish_raw_and_rgb(np.flipud(self.spectrogram.T), self.image_stream_raw, self.image_stream_rgb, header)

        # Perform medfilt
        max_values = np.max(self.spectrogram, axis=1)
        max_indices = np.argmax(self.spectrogram, axis=1)
        max_freqs = self.trimmed_freqs[max_indices]
        filtered_max_freqs = medfilt(max_freqs, kernel_size=self.medfilt_kernel)
        filtered_max_freqs = np.where(filtered_max_freqs == np.min(filtered_max_freqs), np.nan, filtered_max_freqs)

        # Perform min-max normalization
        min_val = self.normalize_min
        max_val = self.normalize_max
        if self.normalize_do:
            normalized_spectrogram = (np.flipud(self.spectrogram.T)
                                      - min_val) / (max_val - min_val)
        else:
            normalized_spectrogram = np.flipud(self.spectrogram.T)

        # Publish preprocessed stream
        self.publish_raw_and_rgb(normalized_spectrogram, self.image_preprocessed_raw, self.image_preprocessed_rgb, header)
        self.publish_spectrogram(normalized_spectrogram, header, self.sample_rate, self.chunk_size)

        # Activity detection & PitchTrack
        self._activity_detection(header, max_values, filtered_max_freqs)

    def _activity_detection(self, header, max_values, filtered_max_freqs):
        """Detect whistle events and, if complete, publish a PitchTrack."""
        # Only proceed if activity detection is enabled
        if not self.run_activity_detection:
            return

        # Compute time axis for each time slice
        times = np.linspace(0, self.window_duration, self.num_chunks)

        current_marker = filtered_max_freqs[-1]
        current_time = time.time()  # Absolute current time in seconds

        if not self.recording and not np.isnan(current_marker):
            self.recording = True
            self.record_start_time = current_time
            self.get_logger().info("Start recording")
            return

        if self.recording and np.isnan(current_marker) and \
                current_time - self.record_start_time > self.record_duration:
            duration = current_time - self.record_start_time
            event_start_rel = self.window_duration - duration
            event_indices = np.where((times >= event_start_rel) &
                                     (max_values > self.secondary_threshold_db))[0]
            recorded_points = list(zip(times[event_indices],
                                       filtered_max_freqs[event_indices]))

            # --- Build and publish PitchTrack ---------------------------
            pt = PitchTrack()
            pt.header = header

            duration_time = RosTime()
            duration_time.sec = int(duration)
            duration_time.nanosec = int((duration - int(duration)) * 1e9)
            pt.duration = duration_time

            pt.peak_frequency_contour = []
            for t_rel, freq in recorded_points:
                sp = SpectralPeak()
                point_time = self.record_start_time + t_rel
                pt_time = RosTime()
                pt_time.sec = int(point_time)
                pt_time.nanosec = int((point_time - int(point_time)) * 1e9)
                sp.stamp = pt_time           # Time sub-message
                sp.peak_frequency = Float32(data=freq)
                pt.peak_frequency_contour.append(sp)

            self.pitch_pub.publish(pt)
            self.get_logger().info("Published PitchTrack with {} points."
                                   .format(len(pt.peak_frequency_contour)))

            # Draw markers & publish pre-processed images ---------------
            viridis = plt.get_cmap('viridis')
            min_val = self.normalize_min
            max_val = self.normalize_max
            if self.normalize_do:
                normalized_spectrogram = (np.flipud(self.spectrogram.T)
                                          - min_val) / (max_val - min_val)
            else:
                normalized_spectrogram = np.flipud(self.spectrogram.T)

            normalized_uint8 = (normalized_spectrogram * 255).astype(np.uint8)
            normalized_uint8 = np.ascontiguousarray(normalized_uint8)

            colored = viridis(normalized_spectrogram)[:, :, :3]
            colored = (colored * 255).astype(np.uint8)

            valid = max_values > self.secondary_threshold_db
            height, width = normalized_uint8.shape
            for t, freq in zip(times[valid], filtered_max_freqs[valid]):
                if np.isnan(freq):
                    continue
                x = int((t / self.window_duration) * width)
                y = height - int(((freq - self.frequency_min) /
                                  (self.frequency_max - self.frequency_min)) * height)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(normalized_uint8, (x, y),
                               radius=2, color=255, thickness=1)
                    cv2.circle(colored, (x, y),
                               radius=2, color=(255, 0, 0), thickness=1)


            # Publish input stream
            self.publish_raw_and_rgb(np.flipud(self.spectrogram.T), self.image_codex_raw, self.image_codex_rgb, header)

            # Publish images marked with codex
            #ros_img = self.bridge.cv2_to_imgmsg(normalized_uint8, encoding="mono8")
            #ros_img.header = header
            #self.image_codex_raw.publish(ros_img)
            #ros_img = self.bridge.cv2_to_imgmsg(colored, encoding="rgb8")
            #ros_img.header = header
            #self.image_codex_rgb.publish(ros_img)

            self.recording = False
            self.record_start_time = None



def main():
    rclpy.init()
    try:
        rclpy.spin(RealTimeSpectrogram())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
