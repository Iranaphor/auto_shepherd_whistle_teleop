import numpy as np
import rclpy
from rclpy.node import Node
from auto_shepherd_msgs.msg import PitchTrack, PitchCode  # Ensure these packages are available in your workspace
from std_msgs.msg import UInt8
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

class PitchDecoder(Node):
    def __init__(self):
        super().__init__('pitch_decoder')

        # Subscriber for the pitch track topic using custom QoS.
        self.subscription = self.create_subscription(
            PitchTrack,
            'pitch_track',
            self.pitch_callback,
            self.get_qos()
        )

        # Publisher for the decoded pitch codes.
        self.code_publisher = self.create_publisher(
            PitchCode,
            'pitch_code',
            self.get_qos()
        )

        # Thresholds for detecting pauses.
        self.pause_threshold = 0.1      # Only gaps longer than this will be marked as a pause

        self.get_logger().info("Pitch Decoder Node Initialized.")

    def get_qos(self):
        """Sets a QoS profile to ensure reliable and latched message delivery."""
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

    def filter_nan_series(self, peaks):
        filtered_peaks = []
        previous_was_nan = False

        for p in peaks:
            # Check if the value is NaN
            if np.isnan(p.peak_frequency.data):
                if not previous_was_nan:
                    # Append the first NaN in a series
                    filtered_peaks.append(p)
                    previous_was_nan = True
                # Else: skip this NaN because it's consecutive
            else:
                filtered_peaks.append(p)
                previous_was_nan = False

        return filtered_peaks

    def pitch_callback(self, msg):
        """
        Processes the incoming PitchTrack message and applies a filter to simplify the output.
        The spectral peak contour is decoded into trend codes ("up", "down", or "pause"),
        where a "pause" is only detected if the time gap between peaks exceeds the long pause threshold.
        Consecutive duplicate codes are filtered out. The final list is then published as a PitchCode message.
        """
        msg.peak_frequency_contour = self.filter_nan_series(msg.peak_frequency_contour)
        peaks = msg.peak_frequency_contour

        from pprint import pprint
        msgtime = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        duration = msg.duration.sec + msg.duration.nanosec * 1e-9
        for i, id in enumerate(peaks[:-1]):
            ptime_c = peaks[i].stamp.sec + peaks[i].stamp.nanosec * 1e-9
            ptime_n = peaks[i+1].stamp.sec + peaks[i+1].stamp.nanosec * 1e-9
            ptime_c_i = ptime_c - msgtime - duration
            ptime_n_i = ptime_n - msgtime - duration
            #print(round(ptime_n_i - ptime_c_i, 2), peaks[i].peak_frequency.data)


        if not peaks:
            self.get_logger().warn("Received PitchTrack message with no spectral peaks.")
            return

        raw_codes = []
        # Use the first peak as a starting point.
        prev_time = peaks[0].stamp.sec + peaks[0].stamp.nanosec * 1e-9
        prev_freq = peaks[0].peak_frequency.data

        for peak in peaks[1:]:
            curr_time = peak.stamp.sec + peak.stamp.nanosec * 1e-9
            curr_freq = peak.peak_frequency.data
            dt = curr_time - prev_time

            # Only consider a pause if the time gap is significantly longer than normal.
            if dt > self.pause_threshold:
                raw_codes.append("pause")

            # Determine inflection based on frequency change.
            if curr_freq > prev_freq:
                raw_codes.append("up")
            elif curr_freq < prev_freq:
                raw_codes.append("down")
            else:
                # If no frequency change, ignore this sample.
                pass

            # Update previous values for next iteration.
            prev_time = curr_time
            prev_freq = curr_freq

        # Filter out consecutive duplicate codes.
        filtered_codes = []
        for code in raw_codes:
        #    if not filtered_codes or code != filtered_codes[-1]:
                filtered_codes.append(code)

        #print(filtered_codes)

        # Convert filtered string codes into numerical codes using the PitchCode constants.
        code_list = []
        for code in filtered_codes:
            if code == "up":
                code_list.append(PitchCode.UP)
            elif code == "down":
                code_list.append(PitchCode.DOWN)
            elif code == "pause":
                code_list.append(PitchCode.PAUSE)
            else:
                self.get_logger().warn(f"Unknown code encountered: {code}")

        # Build the PitchCode message.
        pitch_code_msg = PitchCode()
        pitch_code_msg.header.stamp = self.get_clock().now().to_msg()
        # Since the msg definition requires an array of std_msgs/UInt8,
        # wrap each numerical code into a UInt8 message.
        pitch_code_msg.code = [UInt8(data=c) for c in code_list]

        # Publish the PitchCode message.
        self.code_publisher.publish(pitch_code_msg)
        self.get_logger().info(f"Published Pitch Code: {[c.data for c in pitch_code_msg.code]}")

def main(args=None):
    rclpy.init(args=args)
    node = PitchDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



















