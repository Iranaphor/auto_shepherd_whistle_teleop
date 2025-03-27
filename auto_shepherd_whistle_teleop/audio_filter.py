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
from auto_shepherd_msgs.msg import PitchTrack, SpectralPeak
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time as RosTime

class RealTimeSpectrogram:
    def __init__(self, node, pitch_pub, image_pub, image_stream,
                 sample_rate=44100, chunk_size=1024, window_duration=5,
                 threshold_db=-10, secondary_threshold_db=-5, medfilt_kernel=3,
                 show_display=True):
        """
        node: A ROS2 Node instance for logging and clock.
        pitch_pub: A ROS2 publisher for PitchTrack messages.
        show_display: Boolean flag. If True, the spectrogram display (GUI) is shown.
        """
        self.node = node
        self.pitch_pub = pitch_pub
        self.image_pub = image_pub
        self.image_stream = image_stream
        self.show_display = show_display
        self.bridge = CvBridge()

        # Audio and display parameters
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_duration = window_duration
        self.num_chunks = int(window_duration * sample_rate / chunk_size)
        self.threshold_db = threshold_db
        self.secondary_threshold_db = secondary_threshold_db
        self.medfilt_kernel = medfilt_kernel

        # Compute full FFT frequency bins and create frequency mask (500-5000 Hz)
        full_freqs = np.fft.rfftfreq(chunk_size, d=1/sample_rate)
        self.freq_mask = (full_freqs >= 500) & (full_freqs <= 5000)
        self.trimmed_freqs = full_freqs[self.freq_mask]

        # Initialize spectrogram buffer (rows: time slices, columns: frequency bins)
        self.spectrogram = np.zeros((self.num_chunks, len(self.trimmed_freqs)))

        # Event detection variables
        self.recording = False
        self.record_start_time = None

        # Setup display if required
        if self.show_display:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.node.get_logger().warning(str(status))
        # Process mono audio
        audio_data = indata[:, 0]
        windowed = audio_data * np.hanning(len(audio_data))
        fft_data = np.abs(np.fft.rfft(windowed))
        fft_data = 20 * np.log10(fft_data + 1e-6)
        fft_data = np.where(fft_data < self.threshold_db, 0, fft_data)
        fft_data_trimmed = fft_data[self.freq_mask]
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
        # If display is enabled, update the plot
        if self.show_display:
            self.ax.clear()
            extent = [0, self.window_duration, self.trimmed_freqs[0], self.trimmed_freqs[-1]]
            self.ax.imshow(self.spectrogram.T, origin='lower', aspect='auto',
                           extent=extent, cmap='viridis')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Frequency (Hz)")
            self.ax.set_title("Real-Time Spectrogram (500-5000 Hz) with Argmax")
            valid = max_values > self.secondary_threshold_db
            if np.any(valid):
                self.ax.plot(times[valid], filtered_max_freqs[valid], marker='o',
                             linestyle='-', color='red', markersize=5)

        # Format datastream
        viridis = plt.get_cmap('viridis')
        min_val = np.min(self.spectrogram)
        max_val = np.max(self.spectrogram)

        # Perform min-max normalization
        normalized_spectrogram = (np.flipud(self.spectrogram.T) - min_val) / (max_val - min_val)
        colored_spectrogram = viridis(normalized_spectrogram)[:, :, :3]  # Exclude the alpha channel
        colored_spectrogram = (colored_spectrogram * 255).astype(np.uint8)

        # Convert to ROS2 Image and publish
        ros_image = self.bridge.cv2_to_imgmsg(colored_spectrogram, encoding="rgb8")
        self.image_stream.publish(ros_image)

        # --- Event Detection & PitchTrack Publishing ---
        current_marker = filtered_max_freqs[-1]
        current_time = time.time()  # Absolute current time in seconds

        if not self.recording and not np.isnan(current_marker):
            self.recording = True
            self.record_start_time = current_time
            self.node.get_logger().info("Start recording")
        elif self.recording and np.isnan(current_marker) and current_time - self.record_start_time > 1.5:
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
            now = self.node.get_clock().now().to_msg()
            pt.header = Header()
            pt.header.stamp = now
            pt.header.frame_id = "spectrogram"

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
            min_val = np.min(self.spectrogram)
            max_val = np.max(self.spectrogram)

            # Perform min-max normalization
            normalized_spectrogram = (np.flipud(self.spectrogram.T) - min_val) / (max_val - min_val)
            colored_spectrogram = viridis(normalized_spectrogram)[:, :, :3]  # Exclude the alpha channel
            colored_spectrogram = (colored_spectrogram * 255).astype(np.uint8)

            # Convert to ROS2 Image and publish
            ros_image = self.bridge.cv2_to_imgmsg(colored_spectrogram, encoding="rgb8")
            self.image_pub.publish(ros_image)

            self.recording = False
            self.record_start_time = None
        elif self.recording:
            self.node.get_logger().debug('.')
        return []

    def start(self):
        # Open the audio stream and run the update loop.
        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                            blocksize=self.chunk_size, callback=self.audio_callback):
            if self.show_display:
                plt.tight_layout()
                plt.show()  # This blocks, running in the main thread.
            else:
                # Without display, run update() periodically in a loop.
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
        self.pitch_pub = self.create_publisher(
            PitchTrack,
            'pitch_track',
            self.get_qos()
        )
        self.image_pub = self.create_publisher(
            Image,
            'pitch_image',
            self.get_image_qos()
        )
        self.image_stream = self.create_publisher(
            Image,
            'audio_stream',
            self.get_image_qos()
        )
        # Pass show_display argument here (set to False to disable GUI)
        self.spectro = RealTimeSpectrogram(node=self, pitch_pub=self.pitch_pub,
                                           image_pub=self.image_pub, image_stream=self.image_stream,
                                           sample_rate=44100, chunk_size=1024, window_duration=5,
                                           threshold_db=-10, secondary_threshold_db=-5,
                                           medfilt_kernel=3, show_display=False)

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
