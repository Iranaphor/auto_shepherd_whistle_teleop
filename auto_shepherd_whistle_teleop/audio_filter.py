import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
import threading
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import medfilt

# RealTimeSpectrogram encapsulates the spectrogram functionality.
class RealTimeSpectrogram:
    def __init__(self, sample_rate=44100, chunk_size=1024, window_duration=5,
                 threshold_db=-10, secondary_threshold_db=-5, medfilt_kernel=3):
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

        # Setup plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        # Process mono audio
        audio_data = indata[:, 0]
        # Apply a Hanning window to reduce spectral leakage
        windowed = audio_data * np.hanning(len(audio_data))
        # Compute FFT magnitude for positive frequencies
        fft_data = np.abs(np.fft.rfft(windowed))
        # Convert to dB scale (avoid log(0))
        fft_data = 20 * np.log10(fft_data + 1e-6)
        # Binary thresholding: set values below threshold to 0
        fft_data = np.where(fft_data < self.threshold_db, 0, fft_data)
        # Trim the FFT data to keep only frequencies between 500 and 5000 Hz
        fft_data_trimmed = fft_data[self.freq_mask]
        # Roll the spectrogram buffer and append the new data
        self.spectrogram[:-1, :] = self.spectrogram[1:, :]
        self.spectrogram[-1, :] = fft_data_trimmed

    def update(self, frame):
        self.ax.clear()
        # Define extent: [time_start, time_end, frequency_start, frequency_end]
        extent = [0, self.window_duration, self.trimmed_freqs[0], self.trimmed_freqs[-1]]
        self.ax.imshow(self.spectrogram.T, origin='lower', aspect='auto',
                       extent=extent, cmap='viridis')
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Real-Time Spectrogram (500-5000 Hz) with Argmax")

        # Compute time axis for each time slice
        times = np.linspace(0, self.window_duration, self.num_chunks)
        # For each time slice, compute maximum amplitude and its frequency bin
        max_values = np.max(self.spectrogram, axis=1)
        max_indices = np.argmax(self.spectrogram, axis=1)
        max_freqs = self.trimmed_freqs[max_indices]

        # Apply median filter to smooth frequency markers
        filtered_max_freqs = medfilt(max_freqs, kernel_size=self.medfilt_kernel)
        # Replace markers equal to the minimum value with NaN
        filtered_max_freqs = np.where(filtered_max_freqs == np.min(filtered_max_freqs),
                                      np.nan, filtered_max_freqs)

        # Plot valid markers (time slices where max amplitude exceeds secondary threshold)
        valid = max_values > self.secondary_threshold_db
        if np.any(valid):
            self.ax.plot(times[valid], filtered_max_freqs[valid], marker='o', 
                         linestyle='-', color='red', markersize=5)

        # --- Event Detection ---
        # Use the most recent marker (rightmost time slice) as the current point.
        current_marker = filtered_max_freqs[-1]
        current_time = time.time()  # absolute current time in seconds

        if not self.recording and not np.isnan(current_marker):
            # Detected the start of a new event
            self.recording = True
            self.record_start_time = current_time
            print("Start recording")
        elif self.recording and np.isnan(current_marker) and current_time - self.record_start_time > 1.5:
            # Event finished after at least 1.5 seconds.
            duration = current_time - self.record_start_time
            event_start_rel = self.window_duration - duration
            event_indices = np.where((times >= event_start_rel) & (max_values > self.secondary_threshold_db))[0]
            recorded_points = list(zip(times[event_indices], filtered_max_freqs[event_indices]))
            print("\nEvent completed:")
            print("  Start time: {:.3f} sec".format(self.record_start_time))
            print("  Stop time:  {:.3f} sec".format(current_time))
            print("  Duration:   {:.3f} sec".format(duration))
            print("  Recorded points (timestamp, marker):")
            for t, m in recorded_points:
                print("    ({:.3f}, {:.3f})".format(t, m))
            self.recording = False
            self.record_start_time = None
        elif self.recording:
            print('.', end='', flush=True)

        return []

    def start(self):
        # Open the audio stream and start the plot animation (blocking call)
        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                            blocksize=self.chunk_size, callback=self.audio_callback):
            plt.tight_layout()
            plt.show()

# ROS2 Node wrapping the RealTimeSpectrogram functionality.
class RealTimeSpectrogramNode(Node):
    def __init__(self):
        super().__init__('real_time_spectrogram_node')
        self.get_logger().info('Real Time Spectrogram Node initialized.')
        # Instantiate the RealTimeSpectrogram object with desired parameters.
        self.spectro = RealTimeSpectrogram(sample_rate=44100, chunk_size=1024, window_duration=5,
                                           threshold_db=-10, secondary_threshold_db=-5, medfilt_kernel=3)

    def get_qos(self):
        """Returns a QoS profile for reliable, latched message delivery."""
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

def main():
    rclpy.init()
    node = RealTimeSpectrogramNode()
    # Start the ROS2 spin in a separate thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    # Run the spectrogram (with its matplotlib GUI) in the main thread.
    node.spectro.start()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
