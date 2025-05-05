import os
import yaml
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
import threading
import time
import numpy as np
import sounddevice as sd
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import medfilt

# Import message types
from auto_shepherd_msgs.msg import PitchTrack, SpectralPeak, AudioChunk, Spectrogram
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time as RosTime

class RealTimeSpectrogram:
    def __init__(self, node, config, pub_raw_spectrogram, pitch_pub,
                       image_stream_raw, image_stream_rgb,
                       image_preprocessed_raw, image_preprocessed_rgb):
        """
        node: A ROS2 Node instance for logging and clock.
        pitch_pub: A ROS2 publisher for PitchTrack messages.
        show_display: Boolean flag. If True, the spectrogram display (GUI) is shown.
        """
        self.node = node
        self.image_stream_raw = image_stream_raw
        self.image_stream_rgb = image_stream_rgb
        self.image_preprocessed_raw = image_preprocessed_raw
        self.image_preprocessed_rgb = image_preprocessed_rgb
        self.pitch_pub = pitch_pub
        self.bridge = CvBridge()

        # Audio and display parameters
        self.config = config

        # Audio and display parameters
        p = self.config['audio_preprocessor']['preprocessing']
        c = self.config['codex_detector']['activity_detection']

        # Setup preprocessing details
        self.threshold_db = p['threshold_db']
        self.secondary_threshold_db = p['secondary_threshold_db']

        # Save preprocessing properties
        self.medfilt_do = p['medfilt']['do']
        self.medfilt_kernel = p['medfilt']['kernel']

        self.normalize_do = p['normalize']['do']
        self.normalize_min = p['normalize']['min']
        self.normalize_max = p['normalize']['max']

        self.frequency_crop_do = p['frequency_crop']['do']
        self.frequency_crop_min = p['frequency_crop']['min']
        self.frequency_crop_max = p['frequency_crop']['max']

        # Save activity detection properties
        self.run_activity_detection = c['run']
        self.record_duration = c['record_duration']

        # Setup audio input details
        self.pause = True
        self.resume_mic()

        # Event detection variables
        self.recording = False
        self.record_start_time = None

    def pause_mic(self, msg):
        if not self.pause:
            self.pause = True
            print('mic paused')

            # Audio and display parameters
            s = self.config['audio_preprocessor']['spectrogram']

            # Save input stream properties
            self.sample_rate = msg.sample_rate
            self.chunk_size = msg.frames
            self.window_duration = s['window_duration']
            self.num_chunks = int(s['window_duration'] * self.sample_rate / self.chunk_size)

            # Compute full FFT frequency bins and create frequency mask (500-5000 Hz)
            full_freqs = np.fft.rfftfreq(self.chunk_size, d=1/self.sample_rate)
            if self.frequency_crop_do:
                self.freq_mask = (full_freqs >= self.frequency_crop_min) & (full_freqs <= self.frequency_crop_max)
                self.trimmed_freqs = full_freqs[self.freq_mask]
            else:
                self.trimmed_freqs = full_freqs

            # Initialize spectrogram buffer (rows: time slices, columns: frequency bins)
            self.spectrogram = np.zeros((self.num_chunks, len(self.trimmed_freqs)))

    def resume_mic(self):
        if self.pause:
            # Audio and display parameters
            i = self.config['audio_input_mic']['input_stream']
            s = self.config['audio_preprocessor']['spectrogram']
            # Save input stream properties
            self.sample_rate = i['sample_rate']
            self.chunk_size = i['chunk_size']
            self.window_duration = s['window_duration']
            self.num_chunks = int(s['window_duration'] * i['sample_rate'] / i['chunk_size'])

            # Compute full FFT frequency bins and create frequency mask (500-5000 Hz)
            full_freqs = np.fft.rfftfreq(self.chunk_size, d=1/self.sample_rate)
            if self.frequency_crop_do:
                self.freq_mask = (full_freqs >= self.frequency_crop_min) & (full_freqs <= self.frequency_crop_max)
                self.trimmed_freqs = full_freqs[self.freq_mask]
            else:
                self.trimmed_freqs = full_freqs

            # Initialize spectrogram buffer (rows: time slices, columns: frequency bins)
            self.spectrogram = np.zeros((self.num_chunks, len(self.trimmed_freqs)))

            self.pause = False
            print('mic resumed')


    def audio_callback(self, indata, frames, time_info, status):
        # Called in the background by the while loop in self.start()
        if self.pause:
            if status != "from_file":
                return
        elif status:
            self.node.get_logger().warning(str(status))
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


    def update(self, frame=0):
        # Compute time axis for each time slice
        times = np.linspace(0, self.window_duration, self.num_chunks)
        max_values = np.max(self.spectrogram, axis=1)
        max_indices = np.argmax(self.spectrogram, axis=1)
        max_freqs = self.trimmed_freqs[max_indices]
        filtered_max_freqs = medfilt(max_freqs, kernel_size=self.medfilt_kernel)
        filtered_max_freqs = np.where(filtered_max_freqs == np.min(filtered_max_freqs),
                                      np.nan, filtered_max_freqs)

        # Format datastream
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = "spectrogram"

        viridis = plt.get_cmap('viridis')
        min_val = self.normalize_min
        max_val = self.normalize_max

        # Perform min-max normalization
        if self.normalize_do:
            normalized_spectrogram = (np.flipud(self.spectrogram.T) - min_val) / (max_val - min_val)
        else:
            normalized_spectrogram = np.flipud(self.spectrogram.T)
        normalized_spectrogram_uint8 = (normalized_spectrogram * 255).astype(np.uint8)

        colored_spectrogram = viridis(normalized_spectrogram)[:, :, :3]  # Exclude the alpha channel
        colored_spectrogram = (colored_spectrogram * 255).astype(np.uint8)

        # Convert to ROS2 Image and publish
        ros_image = self.bridge.cv2_to_imgmsg(normalized_spectrogram_uint8, encoding="mono8")
        ros_image.header = header
        self.image_stream_raw.publish(ros_image)
        ros_image = self.bridge.cv2_to_imgmsg(colored_spectrogram, encoding="rgb8")
        ros_image.header = header
        self.image_stream_rgb.publish(ros_image)

        # Only proceed if activity detection is enabled
        if not self.run_activity_detection:
            return []

        # --- Event Detection & PitchTrack Publishing ---
        current_marker = filtered_max_freqs[-1]
        current_time = time.time()  # Absolute current time in seconds

        if not self.recording and not np.isnan(current_marker):
            self.recording = True
            self.record_start_time = current_time
            self.node.get_logger().info("Start recording")
        elif self.recording and np.isnan(current_marker) and current_time - self.record_start_time > self.record_duration:
            duration = current_time - self.record_start_time
            event_start_rel = self.window_duration - duration
            event_indices = np.where((times >= event_start_rel) & (max_values > self.secondary_threshold_db))[0]
            recorded_points = list(zip(times[event_indices], filtered_max_freqs[event_indices]))
            self.node.get_logger().info("\nEvent completed:")
            self.node.get_logger().info("  Start time: {:.3f} sec".format(self.record_start_time))
            self.node.get_logger().info("  Stop time:  {:.3f} sec".format(current_time))
            self.node.get_logger().info("  Duration:   {:.3f} sec".format(duration))
            self.node.get_logger().info("  Recorded points (timestamp, marker):")
            for t, m in recorded_points:
                self.node.get_logger().info("    ({:.3f}, {:.3f})".format(t, m))

            # --- Build and Publish the PitchTrack message ---
            pt = PitchTrack()
            pt.header = header

            # Create a Time message representing the duration of the event
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
                sp.stamp = pt_time  # Assign directly as a Time sub-message
                sp.peak_frequency = Float32(data=freq)
                pt.peak_frequency_contour.append(sp)

            self.pitch_pub.publish(pt)
            self.node.get_logger().info("Published PitchTrack with {} points.".format(len(pt.peak_frequency_contour)))

            # Format datastream
            viridis = plt.get_cmap('viridis')
            min_val = self.normalize_min
            max_val = self.normalize_max

            # Perform min-max normalization
            if self.normalize_do:
                normalized_spectrogram = (np.flipud(self.spectrogram.T) - min_val) / (max_val - min_val)
            else:
                normalized_spectrogram = np.flipud(self.spectrogram.T)

            normalized_spectrogram_uint8 = (normalized_spectrogram * 255).astype(np.uint8)
            normalized_spectrogram_uint8 = np.ascontiguousarray(normalized_spectrogram_uint8)

            colored_spectrogram = viridis(normalized_spectrogram)[:, :, :3]
            colored_spectrogram = (colored_spectrogram * 255).astype(np.uint8)

            # Draw markers on the mono image (white circle with pixel value 255)
            valid = max_values > self.secondary_threshold_db
            height, width = normalized_spectrogram_uint8.shape
            for t, freq in zip(times[valid], filtered_max_freqs[valid]):
                # Check if not NaN
                if np.isnan(freq):
                    continue
                # Convert time and frequency to x/y
                x = int((t / self.window_duration) * width)
                y = height - int(((freq - self.frequency_min) / (self.frequency_max - self.frequency_min)) * height)
                # Plot circles to raw and rgb images if within image
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(normalized_spectrogram_uint8, (x, y), radius=2, color=255, thickness=1)
                    cv2.circle(colored_spectrogram, (x, y), radius=2, color=(255, 0, 0), thickness=1)

            # Convert to ROS2 Image and publish
            ros_image = self.bridge.cv2_to_imgmsg(normalized_spectrogram_uint8, encoding="mono8")
            ros_image.header = header
            self.image_pub_raw.publish(ros_image)
            ros_image = self.bridge.cv2_to_imgmsg(colored_spectrogram, encoding="rgb8")
            ros_image.header = header
            self.image_pub_rgb.publish(ros_image)

            self.recording = False
            self.record_start_time = None
        elif self.recording:
            self.node.get_logger().debug('.')
        return []

    def start(self):
        # Open the audio stream and run the update loop.
        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                            blocksize=self.chunk_size, callback=self.audio_callback):
            try:
                while rclpy.ok():
                    self.update()
                    time.sleep(0.05)  # 50 ms interval
            except KeyboardInterrupt:
                pass

# ROS2 Node wrapping the RealTimeSpectrogram functionality.
class RealTimeSpectrogramNode(Node):
    def __init__(self):
        super().__init__('real_time_spectrogram_node')
        self.get_logger().info('Real Time Spectrogram Node initialized.')

        # get topic data from config
        self.config_file = os.getenv('WHISTLE_CONF')
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        p = self.config['audio_preprocessor']['topics']
        c = self.config['codex_detector']['topics']

        # Subscriber for incoming audio from input node (raw audio chunk)
        self.subscription = self.create_subscription(AudioChunk, p['input']['audio'], self.audio_input_cb, 10)

        # Publisher for spectrogram (raw)
        self.pub_raw_spectrogram = self.create_publisher(Spectrogram, p['output']['raw'], self.get_image_qos())

        # Publishers for strean (raw/rgb)
        self.image_stream_raw = self.create_publisher(Image, p['visual']['st_raw'], self.get_image_qos())
        self.image_stream_rgb = self.create_publisher(Image, p['visual']['st_rgb'], self.get_image_qos())

        # Publishers for preprocessed stream (raw/rgb)
        self.image_preprocessed_raw = self.create_publisher(Image, p['visual']['pr_raw'], self.get_image_qos())
        self.image_preprocessed_rgb = self.create_publisher(Image, p['visual']['pr_rgb'], self.get_image_qos())

        # Publisher for detected activity (codex)
        self.pitch_pub = self.create_publisher(PitchTrack, c['output']['codex'], self.get_qos())

        # Pass show_display argument here (set to False to disable GUI)
        self.spectro = RealTimeSpectrogram(node = self,
                                           config = self.config,
                                           pitch_pub = self.pitch_pub,
                                           pub_raw_spectrogram = self.pub_raw_spectrogram,
                                           image_stream_raw = self.image_stream_raw,
                                           image_stream_rgb = self.image_stream_rgb,
                                           image_preprocessed_raw = self.image_preprocessed_raw,
                                           image_preprocessed_rgb = self.image_preprocessed_rgb)

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

    def audio_input_cb(self, msg):
        #print(len(msg.data), msg.frames, msg.channels, msg.sample_rate)
        if not msg.data:
            self.spectro.resume_mic()
            return
        if not self.spectro.pause:
            self.spectro.pause_mic(msg)

        # Called as callback from audio_input.py
        indata = np.asarray(msg.data, dtype=np.float32)\
                    .reshape((msg.frames, msg.channels))
        frames = msg.frames
        # fabricate minimal time_info / status if you need them
        time_info = {'input_buffer_adc_time': msg.header.stamp.sec +
                                             msg.header.stamp.nanosec * 1e-9}
        status = "from_file"

        self.spectro.audio_callback(indata, frames, time_info, status)


def main():
    rclpy.init()
    node = RealTimeSpectrogramNode()
    # Start the ROS2 spin in a separate thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    # Run the spectrogram (display toggled by show_display flag)
    node.spectro.start()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
