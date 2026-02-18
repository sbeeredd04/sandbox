#!/usr/bin/env python3
"""
interactive_string_pub.py

Continuously publishes std_msgs/String on a topic.
Type a new line and press Enter at any time to update the message.
Type :q and press Enter to quit.

Usage:
  ./interactive_string_pub.py --topic /spinflow/goal_cmd_override --rate 2 \
      --initial "Go to the next waypoint" --latch
"""

import argparse
import threading
import time

import rospy
from std_msgs.msg import String

# Shared state
_state_lock = threading.Lock()
_current_text = ""
_stop = threading.Event()

def input_worker():
    """Blocking reader: waits for a line from stdin and updates the message."""
    global _current_text
    print("[interactive] Type a new message and press Enter (':q' to quit).")
    while not _stop.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            _stop.set()
            return
        if line.strip() == ":q":
            _stop.set()
            return
        with _state_lock:
            _current_text = line  # keep exactly what user typed (including spaces)
        print(f"[interactive] Updated message to: {repr(line)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/spinflow/goal_cmd_override",
                        help="Topic to publish std_msgs/String on")
    parser.add_argument("--rate", type=float, default=1.0, help="Publish rate (Hz)")
    parser.add_argument("--initial", default="Go to the next waypoint",
                        help="Initial message text")
    parser.add_argument("--queue_size", type=int, default=10)
    parser.add_argument("--latch", action="store_true", help="Latch the publisher")
    args = parser.parse_args()

    global _current_text
    with _state_lock:
        _current_text = args.initial

    rospy.init_node("interactive_string_pub", anonymous=True)
    pub = rospy.Publisher(args.topic, String, queue_size=args.queue_size, latch=args.latch)
    rate = rospy.Rate(args.rate)

    # Start the input thread
    t = threading.Thread(target=input_worker, daemon=True)
    t.start()

    print(f"[publisher] Publishing on {args.topic} at {args.rate} Hz (latch={args.latch}).")
    print(f"[publisher] Initial message: {repr(_current_text)}")

    try:
        while not rospy.is_shutdown() and not _stop.is_set():
            with _state_lock:
                msg = String(data=_current_text)
            pub.publish(msg)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        time.sleep(0.05)
        print("\n[publisher] Exiting.")

if __name__ == "__main__":
    main()
