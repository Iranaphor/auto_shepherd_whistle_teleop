#!/usr/bin/env python3
#
# audio_input_mic.py  –  publish live microphone audio as a stream
#
#   publishes :  /input/audio  (auto_shepherd_msgs/AudioChunk)
#
#   Configuration is read from the same YAML passed via WHISTLE_CONF, but under
#   the key  audio_input_mic:                    e.g. in whistle.yaml
#
#   audio_input_mic:
#     input_stream:
#       sample_rate     : 48000      # [Hz]
#       chunk_size      : 1024       # [frames]
#       window_duration : 2.0        # [s]  – used only for “flush–silence”
#       channels        : 1
#     topics:
#       output:
#         audio : "/input/audio"
#

import os
import threading
import yaml
import numpy as np
import sounddevice as sd                # pip install sounddevice
import rclpy
from rclpy.node import Node
from auto_shepherd_msgs.msg import AudioChunk   # generated interface


class MicrophoneInput(Node):
    """
    Continuously capture audio from the default system microphone and publish it
    in fixed-size chunks on the same ROS 2 topic used by *audio_input.py*.
    """

    def __init__(self):
        super().__init__('audio_input_mic')

        # Load config file
        self.config_file = os.getenv('WHISTLE_CONF')
        if not self.config_file or not os.path.isfile(self.config_file):
            raise RuntimeError('Environment variable WHISTLE_CONF must point to a YAML file')
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        s = config['audio_input_mic']['input_stream']
        t = config['audio_input_mic']['topics']

        # Load stream information
        self.sample_delay    = int(s['sample_delay'])
        self.sample_rate     = int(s['sample_rate'])
        self.chunk_size      = int(s['chunk_size'])
        self.channels        = 1

        # Configure publisher
        self.pub = self.create_publisher(AudioChunk, t['output']['audio'], 10)

        # Configure audio capture
        self.stream_thread = threading.Thread(target=self.run_stream, daemon=True)
        self.stream_thread.start()

        # ── New: band-edge metadata (constant for this mic) ───────────────
        # If the mic or driver already applies a known band-pass filter,
        # replace 0 / Nyquist with that narrower range.
        self.frequency_min       = 0                    # Hz
        self.frequency_max       = self.sample_rate // 2  # Nyquist
        # For raw audio we keep the “row-0 is LOW frequency” convention
        # used elsewhere in the pipeline:
        self.frequency_row0_high = False

        self.get_logger().info(f'MicrophoneInput ready (rate={self.sample_rate} Hz, chunk={self.chunk_size} frames)')


    def run_stream(self):
        """Open the microphone and publish chunks until rclpy shuts down."""
        def callback(indata, frames, time_info, status):
            if status:
                self.get_logger().warn(str(status))
            # indata has shape (frames, channels); ensure float32
            chunk = indata.astype(np.float32)
            self.publish_chunk(chunk)

        with sd.InputStream(channels=self.channels, samplerate=self.sample_rate, blocksize=self.chunk_size, dtype='float32', callback=callback):
            try:
                while rclpy.ok():
                    sd.sleep(self.sample_delay)       # Sleep a bit; audio arrives via callback
            except KeyboardInterrupt:
                pass

    def publish_chunk(self, chunk, *, empty: bool = False):
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sample_rate  = self.sample_rate
        msg.frames       = self.chunk_size
        msg.channels     = self.channels
        if empty:
            msg.data = []
        else:
            msg.data = chunk.flatten().tolist()
        msg.frequency_min       = self.frequency_min
        msg.frequency_max       = self.frequency_max
        msg.frequency_row0_high = self.frequency_row0_high # False → low→high

        self.pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(MicrophoneInput())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
