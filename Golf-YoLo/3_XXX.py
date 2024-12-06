import os
import time
from datetime import datetime
import argparse
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from collections import deque
from helpers import start_camera, apply_zoom, resize_frame, detect_person, smooth_detections, create_folder_delete_if_exists
from helpers import FPS, ZOOM, WINDOW_SIZE, CONFIDENCE_THRESHOLD

parser = argparse.ArgumentParser(description="Real-time person detection with smoothing.")
parser.add_argument('-N', '-n', type=int, default=-1, help='Maximum number of images to process (default: -1 (no limit))')
parser.add_argument('-R', '-r', type=str, default="640x480", help='Image resolution (default: 640x480)')
parser.add_argument('-Z', '-z', type=float, default=ZOOM, help='Zoom value (default: 1.0)')
parser.add_argument('-FPS', '-fps', type=float, default=FPS, help='Frames per second (default: 0.25 | 4 sec de sleep)')
parser.add_argument('-WS', '-ws', type=int, default=WINDOW_SIZE, help='Window size for smoothing detection must be odd (default: 11)')
parser.add_argument('-S', '-s', action='store_true', help='To save the images for debug')
parser.add_argument('-A', '-a', action='store_true', help='Prompts the user before creating/erasing the save folder')
parser.add_argument('-V', '-verbose', action='store_true', help='Enables debug prints')
parser.add_argument('-C', '-c', type=float, default=CONFIDENCE_THRESHOLD, help='Confidence threshold for person detection')
args = parser.parse_args()

fps = args.FPS
n_max = args.N
width, height = map(int, args.R.split('x')) 
resolution = (width, height)
zoom = args.Z
window_size = args.WS
require_user = not args.A
save_figs = args.S
verbose = args.V
confidence = args.C

if verbose:
    print(f"FPS: {fps}")
    print(f"N: {n_max}")
    print(f"Resolution: {width}x{height}")
    print(f"Zoom: {zoom}")
    print(f"Window Size: {window_size}")
    print(f"Require user: {require_user}")
    print(f"Save figs: {save_figs}")
    print(f"Using confidence threshold: {confidence}")

output_dir = './detect_images'
if save_figs:
    create_folder_delete_if_exists(output_dir, require_user)

detection_buffer = deque(maxlen=window_size)
person_window_active = False
person_window_start_time = None
person_window_end_time = None

camera = Picamera2()
start_camera(camera, resolution)

#model = YOLO('yolov8n.pt', task="detect")
model = YOLO("yolov8n_ncnn_model", task="detect")

if n_max > 0:
    print(f"Processing maximum {n_max} images...")

print(f"Starting real-time person detection with resolution {resolution} and zoom factor {zoom} fps {fps} window size {window_size}...")

counter = 0
try:
    while True:
        start_time = time.time()
        if verbose:
            print(f"Starting image {counter} {datetime.now()}")

        frame = camera.capture_array()
        zoomed_frame = apply_zoom(frame, zoom)
        resized_frame = resize_frame(zoomed_frame, resolution)
        if verbose:
            print(f"Finish image capture zoom resize {counter} {time.time() - start_time}")

        person_detected = detect_person(model, resized_frame, confidence) #boolean
        if verbose:
            print(f"Finished detection {counter} {time.time() - start_time} {person_detected}")

        if save_figs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            output_file = os.path.join(output_dir, f'img_{timestamp}_{counter}_{int(person_detected)}.jpg')
            cv2.imwrite(output_file, cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            if verbose:
                print(f"Finished saving image {counter} {time.time() - start_time}")

        detection_buffer.append((person_detected, datetime.now()))
        if verbose:
            print(f"Buffer appended {counter} {time.time() - start_time} {len(detection_buffer)}")

        if len(detection_buffer) == 11:
            # Perform smoothing on the last 11 detections
            detection_values = [d[0] for d in detection_buffer]
            if verbose:
                print(f"Starting smoothing on buffer {counter} {time.time() - start_time}")
            smoothed_result = smooth_detections(detection_values)
            if verbose:
                print(f"Finished smoothing on buffer {counter} {time.time() - start_time}")
            smoothed_detection_time = detection_buffer[window_size // 2][1]  # Time corresponding to the center of window (6th image)
            # Process smoothed result
            if smoothed_result:
                if not person_window_active:
                    person_window_active = True
                    person_window_start_time = smoothed_detection_time
                    print(f"Person window started at {person_window_start_time}")
            else:
                if person_window_active:
                    person_window_active = False
                    person_window_end_time = smoothed_detection_time
                    person_window_duration = person_window_end_time - person_window_start_time
                    duration_seconds = person_window_duration.total_seconds()
                    if duration_seconds >= 5 * 60: # Time threshold as alert for now
                        print(f"ALERT: Person detected for more than 5 minutes from {person_window_start_time} to {person_window_end_time} (duration: {person_window_duration})")
                    else:
                        print(f"Person window from {person_window_start_time} to {person_window_end_time} (duration: {person_window_duration})")

                    person_window_start_time = None
                    person_window_end_time = None

        end_time = time.time()
        elapsed_time = end_time - start_time
        if verbose:
            print(f"Finished one image {counter} {elapsed_time}")
        sleep_time = max(0, 1/fps - elapsed_time)
        if verbose:
            print(f"Going to sleep {counter} {sleep_time}")
        time.sleep(sleep_time)
        counter = counter + 1
        if n_max > 0 and counter >= n_max:
                print("Ending...")
                break


except KeyboardInterrupt:
    print("Stopping real-time person detection...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    camera.stop()
    # If a person window is active, process it before exiting
    if person_window_active:
        person_window_end_time = datetime.now()
        duration = person_window_end_time - person_window_start_time
        duration_seconds = duration.total_seconds()
        if duration_seconds >= 5 * 60:
            print(f"ALERT: Person detected for more than 5 minutes from {person_window_start_time} to {person_window_end_time} (duration: {duration})")
        else:
            print(f"Person window from {person_window_start_time} to {person_window_end_time} (duration: {duration})")
    print(f"Processed {counter} images")
