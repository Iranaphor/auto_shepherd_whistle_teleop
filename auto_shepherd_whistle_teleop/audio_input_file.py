#!/usr/bin/env python3
#
# audio_input.py  –  publish any audio/video file as a live-stream
#
#   subscribes:  ~/filepath    (std_msgs/String, absolute path)
#   publishes :  ~/audio       (auto_shepherd_msgs/AudioChunk)
#
# Supported formats: wav, mp3, m4a, mp4 (via FFmpeg), plus anything FFmpeg knows.

import os
import time
import threading
import yaml
import numpy as np

import librosa                           # pip install librosa  (needs ffmpeg)
import soundfile as sf                   # pip install soundfile

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from auto_shepherd_msgs.msg import AudioChunk


class AudioInput(Node):

    def __init__(self):
        super().__init__('audio_input')

        # --------------------------------------------------
        # 1.  Read YAML configuration
        # --------------------------------------------------
        self.config_file = os.getenv('WHISTLE_CONF')
        if not self.config_file or not os.path.isfile(self.config_file):
            raise RuntimeError(
                'Environment variable WHISTLE_CONF must point to a YAML file')

        with open(self.config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        a = cfg['audio_input_file']
        self.sample_rate     = int(a['input_stream']['sample_rate'])
        self.chunk_size      = int(a['input_stream']['chunk_size'])
        self.window_duration = float(a['input_stream']['flush_duration'])
        self.window_chunks   = int(self.window_duration * self.sample_rate / self.chunk_size)

        # --------------------------------------------------
        # 2.  ROS entities
        # --------------------------------------------------
        self.pub  = self.create_publisher(AudioChunk,
                                          a['topics']['output']['audio'],
                                          10)
        self.sub  = self.create_subscription(String,
                                             a['topics']['input']['filepath'],
                                             self._filepath_cb,
                                             10)

        self.get_logger().info(
            f'AudioInput ready (rate={self.sample_rate} Hz, '
            f'chunk={self.chunk_size} frames)')

        # Only one active playback at a time
        self._playback_thread = None


    # ======================================================
    #                 CALLBACKS / HELPERS
    # ======================================================
    def _filepath_cb(self, msg: String):
        """Spawn a worker thread that publishes the requested file."""
        path = os.path.expanduser(msg.data.strip())

        if not os.path.isfile(path):
            self.get_logger().error(f'File not found: {path}')
            return

        if self._playback_thread and self._playback_thread.is_alive():
            #self.get_logger().warn('Playback already running – ignored path')
            return

        self._playback_thread = threading.Thread(
            target=self._stream_file, args=(path,), daemon=True)
        self._playback_thread.start()

    # ------------------------------------------------------------------
    #                        Audio decoding
    # ------------------------------------------------------------------
    def _load_any(self, path: str) -> tuple[np.ndarray, int]:
        """
        Decode *any* audio/video file into a float32 numpy array.

        Returns
        -------
        data : shape (frames, channels) float32
        sr   : sample-rate of the file
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.wav':
            # soundfile gives (frames, channels) already
            data, sr = sf.read(path, dtype='float32', always_2d=True)
        else:
            # librosa/audioread + FFmpeg: (channels, frames) → T to match above
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:            # handle mono
                data = data[np.newaxis, :]
            data = data.T.astype(np.float32)
        return data, sr

    # ──────────────────────────────────────────────────────────────────
    #                  Core: publish file then flush silence
    # ──────────────────────────────────────────────────────────────────
    def _stream_file(self, path: str):
        self.get_logger().info(f'Streaming {path}')

        data, sr = self._load_any(path)

        # Resample if input SR differs from pipeline SR
        if sr != self.sample_rate:
            data = librosa.resample(data,                 # no transpose
                orig_sr=sr,
                target_sr=self.sample_rate,
                axis=0)               # time axis
            sr = self.sample_rate

        frequency_min = 0            # PCM files start at DC
        frequency_max = sr // 2      # Nyquist = half the SR
        frequency_row0_high = False  # row-0 will be the LOWEST freq
        channels   = data.shape[1]
        total_len  = data.shape[0]
        idx        = 0

        # Helper to send one chunk
        def publish_chunk(chunk: np.ndarray, empty=False):
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.sample_rate  = self.sample_rate
            msg.frames       = self.chunk_size
            msg.channels     = channels
            msg.data         = [] if empty else chunk.flatten().tolist()
            msg.frequency_min       = frequency_min
            msg.frequency_max       = frequency_max
            msg.frequency_row0_high = frequency_row0_high
            self.pub.publish(msg)

        # ── Main streaming loop ────────────────────────────────────────
        while rclpy.ok() and idx < total_len:
            chunk = data[idx: idx + self.chunk_size]
            if chunk.shape[0] < self.chunk_size:
                pad = np.zeros((self.chunk_size - chunk.shape[0],
                                channels), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            publish_chunk(chunk)
            if idx%100 == 0:
                progress=(idx/total_len)
                print('published chunk, progress: |'+ '#'*int(30*(progress))+' '*int(30*(1-progress))+f'| {round(progress,2)}%')
            time.sleep(self.chunk_size / self.sample_rate)
            idx += self.chunk_size
        print('published chunk, progress: |'+ '#'*int(30)+' '*int(0)+f'| {round(progress,2)}%')

        # ── Flush silence: one complete sliding-window duration ────────
        zero_chunk = np.zeros((self.chunk_size, channels), dtype=np.float32)
        for _ in range(self.window_chunks):
            if not rclpy.ok():
                break
            publish_chunk(zero_chunk)
            time.sleep(self.chunk_size / self.sample_rate)

        self.get_logger().info('Finished streaming (silence flushed)')

        # Empty message to signal completion
        publish_chunk([], empty=True)
        self.get_logger().info('Finished streaming')


def main():
    rclpy.init()
    try:
        rclpy.spin(AudioInput())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

