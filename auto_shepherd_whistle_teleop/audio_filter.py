import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import medfilt

# Audio and spectrogram parameters
sample_rate = 44100      # Audio sample rate (Hz)
chunk_size = 1024        # Number of samples per audio chunk
window_duration = 5      # Duration (in seconds) for the spectrogram display
num_chunks = int(window_duration * sample_rate / chunk_size)  # Number of time slices

# Compute full FFT frequency bins for a given chunk size
full_freqs = np.fft.rfftfreq(chunk_size, d=1/sample_rate)
# Create a boolean mask to select frequencies between 500 and 5000 Hz
freq_mask = (full_freqs >= 500) & (full_freqs <= 5000)
# Trimmed frequency bins for the spectrogram and plotting
trimmed_freqs = full_freqs[freq_mask]

# Initialize spectrogram with only the trimmed frequency bins
spectrogram = np.zeros((num_chunks, len(trimmed_freqs)))

# Set the dB threshold (all values below this will be zeroed)
THRESHOLD_DB = -10
# Secondary threshold for plotting the argmax marker
SECONDARY_THRESHOLD_DB = -5

# Global variables for event detection
recording = False
record_start_time = None

def audio_callback(indata, frames, time_info, status):
    global spectrogram
    if status:
        print(status)
    # Use only the first channel (mono)
    audio_data = indata[:, 0]
    # Apply a Hanning window to reduce spectral leakage
    windowed = audio_data * np.hanning(len(audio_data))
    # Compute FFT and take the magnitude (only positive frequencies)
    fft_data = np.abs(np.fft.rfft(windowed))
    # Convert to dB scale (adding a small constant to avoid log(0))
    fft_data = 20 * np.log10(fft_data + 1e-6)
    # Apply binary thresholding: set values below THRESHOLD_DB to 0
    fft_data = np.where(fft_data < THRESHOLD_DB, 0, fft_data)
    # Trim the FFT data to keep only frequencies between 500 and 5000 Hz
    fft_data_trimmed = fft_data[freq_mask]
    
    # Roll the spectrogram so that new data appears on the right
    spectrogram[:-1, :] = spectrogram[1:, :]
    spectrogram[-1, :] = fft_data_trimmed

# Set up the figure and axis for the spectrogram
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    global recording, record_start_time
    ax.clear()
    # Define extent: [time_start, time_end, frequency_start, frequency_end]
    extent = [0, window_duration, trimmed_freqs[0], trimmed_freqs[-1]]
    # Display the spectrogram; note the transpose to have time on x and frequency on y
    ax.imshow(spectrogram.T, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Real-Time Spectrogram (500-5000 Hz) with Argmax")
    
    # Compute the time coordinates for each time slice (relative to the window)
    times = np.linspace(0, window_duration, num_chunks)
    # For each time slice, compute the maximum amplitude and its corresponding frequency bin
    max_values = np.max(spectrogram, axis=1)
    max_indices = np.argmax(spectrogram, axis=1)
    # Convert bin indices to actual frequencies using the trimmed frequency array
    max_freqs = trimmed_freqs[max_indices]
    
    # Apply a median filter with a kernel width of 3 to smooth the frequency markers
    filtered_max_freqs = medfilt(max_freqs, kernel_size=3)
    # Replace any marker equal to the minimum of the filtered values with NaN
    filtered_max_freqs = np.where(filtered_max_freqs == np.min(filtered_max_freqs),
                                  np.nan, filtered_max_freqs)
    
    # Plot markers for each time slice where the maximum amplitude exceeds the secondary threshold
    valid = max_values > SECONDARY_THRESHOLD_DB
    if np.any(valid):
        ax.plot(times[valid], filtered_max_freqs[valid], marker='o', linestyle='-', color='red', markersize=5)
    
    # --- Event Detection ---
    # Use the most recent marker (rightmost time slice) as the current point.
    current_marker = filtered_max_freqs[-1]
    current_time = time.time()  # absolute current time in seconds

    if not recording and not np.isnan(current_marker):
        # Detected the start of a new event
        recording = True
        record_start_time = current_time
        print("Start recording")
    elif recording and np.isnan(current_marker) and current_time - record_start_time > 1.5:
        # The event has ended after at least 1.5 seconds.
        duration = current_time - record_start_time
        # Determine the relative time at which the event started in the spectrogram.
        # (Assuming the event started duration seconds ago, relative to the right edge.)
        event_start_rel = window_duration - duration
        # Select indices corresponding to times after the event started and that are valid.
        event_indices = np.where((times >= event_start_rel) & (max_values > SECONDARY_THRESHOLD_DB))[0]
        # Assemble the recorded points from the filtered markers
        recorded_points = list(zip(times[event_indices], filtered_max_freqs[event_indices]))
        print("\nEvent completed:")
        print("  Start time: {:.3f} sec".format(record_start_time))
        print("  Stop time:  {:.3f} sec".format(current_time))
        print("  Duration:   {:.3f} sec".format(duration))
        print("  Recorded points (timestamp, marker):")
        for t, m in recorded_points:
            print("    ({:.3f}, {:.3f})".format(t, m))
        # Reset event detection variables
        recording = False
        record_start_time = None
    elif recording:
        # Continue recording the event (no manual point appending).
        print('.', end='', flush=True)

    return []

# Create an animation that updates the spectrogram plot
ani = FuncAnimation(fig, update, interval=50, blit=False)

# Open the input stream and start the real-time update loop
with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=chunk_size, callback=audio_callback):
    plt.tight_layout()
    plt.show()
