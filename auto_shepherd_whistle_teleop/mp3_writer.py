#!/usr/bin/env python3
import os
import tempfile
import uuid
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import yaml
import librosa
import soundfile as sf
from pydub import AudioSegment

from auto_shepherd_msgs.msg import SpectrogramDetection
from ament_index_python.packages import get_package_prefix

DEFAULT_DB_RANGE = (-80.0, 0.0)

class SpectrogramToMp3(Node):
    def __init__(self):
        super().__init__('spectrogram_to_mp3')
        self.bridge = CvBridge()

        # ── Load YAML config ──
        config_path = os.getenv('WHISTLE_CONF')
        if not config_path:
            raise RuntimeError("WHISTLE_CONF environment variable not set")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            yaml_obj = yaml.safe_load(f)
            self.config = yaml_obj['mp3_writer']

        # ── Parse config fields ──
        input_stream = self.config['input_stream']
        topics = self.config['topics']
        samples = self.config['samples']

        self.sample_rate = int(input_stream['sample_rate'])
        self.chunk_size = int(input_stream['chunk_size'])
        self.window_duration = float(input_stream['window_duration'])

         # Set output directory
        self.output_dir = Path(samples['absolute_dir'])
        if not self.output_dir:
            pkg = samples['src_directory_filepath']['package']
            subpath = samples['src_directory_filepath']['subpath']
            install_prefix = get_package_prefix(pkg)
            base_ws = install_prefix.split('install')[0].strip()
            cmd = ["colcon", "list", "--packages-select", pkg, "--base-paths", base_ws, "-p"]
            src_path = subprocess.check_output(cmd, text=True).strip()
            self.output_dir = Path(src_path) / subpath
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Set up subscription ──
        topic_name = topics['input']['class']
        self.subscription = self.create_subscription(
            SpectrogramDetection,
            topic_name,
            self._spectrogram_cb,
            10
        )

        self.get_logger().info(f"MP3 writer active. Writing to: {self.output_dir}")

    def _spectrogram_cb(self, msg: SpectrogramDetection):
        try:
            # Load parameters from message or default config
            sr = int(msg.sample_rate) if msg.sample_rate > 0 else self.sample_rate
            hop_length = int(msg.hop_length) if msg.hop_length > 0 else self.chunk_size
            n_fft = int(sr * msg.window_duration) if msg.window_duration > 0 else int(sr * self.window_duration)

            # Convert image to magnitude array
            cv_img = self.bridge.imgmsg_to_cv2(msg.raw_spectrogram, desired_encoding='mono8')
            mag_img = cv2.normalize(cv_img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            mag_img = np.flipud(mag_img)
            magnitude = mag_img.T

            db_min, db_max = DEFAULT_DB_RANGE
            magnitude_db = magnitude * (db_max - db_min) + db_min
            magnitude_lin = librosa.db_to_amplitude(magnitude_db)

            # Griffin-Lim reconstruction
            waveform = librosa.griffinlim(
                magnitude_lin,
                n_iter=60,
                hop_length=hop_length,
                win_length=n_fft
            )

            # Save as MP3
            tmp_wav = Path(tempfile.gettempdir()) / f'{uuid.uuid4()}.wav'
            sf.write(tmp_wav, waveform, sr)

            mp3_name = f"{msg.classification}_{uuid.uuid4()}.mp3"
            mp3_path = self.output_dir / mp3_name
            AudioSegment.from_wav(tmp_wav).export(mp3_path, format="mp3")
            tmp_wav.unlink(missing_ok=True)

            self.get_logger().info(f"Saved MP3: {mp3_path}")
        except Exception as e:
            self.get_logger().error(f"Error reconstructing MP3: {e}", exc_info=True)

def main():
    rclpy.init()
    node = SpectrogramToMp3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
