import cv2
import time
import os
import csv
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO
import argparse
from helpers import PERSON_LABEL, FPS, ZOOM, WINDOW_SIZE, BITRATE, CONFIDENCE_THRESHOLD
from helpers import apply_zoom, resize_frame, create_folder_delete_if_exists, start_camera
from helpers import filter_hsv_and_largest_component, save_hsv_plots
import numpy as np
import threading

parser = argparse.ArgumentParser(description="Real-time person detection with preview.")
parser.add_argument('-N', '-n', type=int, default=-1, help='Maximum number of images to process (default: -1 (no limit))')
parser.add_argument('-R', '-r', type=str, default="640x480", help='Image resolution (default: 640x480)')
parser.add_argument('-Z', '-z', type=float, default=ZOOM, help='Zoom value (default: 1.0)')
#parser.add_argument('-B', '-b', type=int, default=BITRATE, help='Camera bitrate (default: 8)')
parser.add_argument('-SL', '-sl', type=float, default=1.0/FPS, help='Sleep between captures (default: 4.0)')
#parser.add_argument('-WS', '-ws', type=int, default=WINDOW_SIZE, help='Window size for smoothing detection must be odd (default: 11)')
parser.add_argument('-S', '-s', action='store_true', help='To save the images for debug')
parser.add_argument('-A', '-a', action='store_true', help='Prompts the user before creating/erasing the save folder')
parser.add_argument('-C', '-c', type=float, default=CONFIDENCE_THRESHOLD, help='Confidence threshold for person detection (default: 0.5)')
parser.add_argument('-V', '-verbose', action='store_true', help='Enables debug prints')
args = parser.parse_args()

model = YOLO("yolov8n_ncnn_model", task="detect")

fps = 1.0/args.SL
#bitrate = args.B
n_max = args.N
width, height = map(int, args.R.split('x')) 
current_resolution = (width, height)
current_zoom = args.Z
#window_size = args.WS
require_user = not args.A
save_figs = args.S
confidence_threshold = args.C
verbose = args.V

if verbose:
    print(f"Sleep: {1.0/fps}")
    print(f"N: {n_max}")
    print(f"Resolution: {width}x{height}")
    #print(f"Bitrate: {bitrate}")
    print(f"Zoom: {current_zoom}")
    #print(f"Window Size: {window_size}")
    print(f"Require user: {require_user}")
    print(f"Save figs: {save_figs}")
    print(f"Using confidence threshold: {confidence_threshold}")

output_dir = './preview_images'
create_folder_delete_if_exists(output_dir, require_user)
log_file = os.path.join(output_dir, f'log.csv')
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Confidence', 'BoundingBox'])

camera = Picamera2()
start_camera(camera, current_resolution)

counter = 0
h_low, h_high = 0, 255
s_low, s_high = 0, 255
v_low, v_high = 0, 255
stop_flag = False
mode_hsv = True
hsv_mask = np.zeros((current_resolution[1], current_resolution[0]), dtype=np.uint8)

def display_persons():
    global stop_flag, mode_hsv, counter, hsv_mask, current_zoom, current_resolution, n_max, confidence_threshold, save_figs

    #threshold = 0.5

    while not stop_flag and not mode_hsv and (n_max == -1 or counter < n_max):

        start_time = time.time()
        
        frame = camera.capture_array()
        zoomed_frame = apply_zoom(frame, current_zoom)
        resized_frame = resize_frame(zoomed_frame, current_resolution)

        results = model.predict(resized_frame, classes=[PERSON_LABEL], verbose=False, conf=confidence_threshold)
        result = results[0]
        if verbose:
            print(f"Detected classes: {result.boxes.cls.int().tolist()}")
            print(f"Confidence scores: {result.boxes.conf.tolist()}")
            print(f"Number of detections: {len(result.boxes)}")
        person_detected = False
        if len(result.boxes) > 0:
            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.int().tolist()
            for class_id, conf_score in zip(class_ids, confidences):
                if verbose:
                    print(f"Class ID: {class_id}, Confidence: {conf_score}")
                if class_id == PERSON_LABEL and conf_score >= confidence_threshold:
                    person_detected = True
                    break
        if verbose:
            print(f"Person detected: {person_detected}")

        boxes_inside_green = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        for result in results: # technically should only contains results[0] as predict is done on the person class
            for box in result.boxes:
                conf_score = float(box.conf)
                if conf_score < confidence_threshold:
                    continue
                bbox = [int(coord) for coord in box.xyxy[0]]
                
                # check if box inside green
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, current_resolution[0]-1))
                y1 = max(0, min(y1, current_resolution[1]-1))
                x2 = max(0, min(x2, current_resolution[0]-1))
                y2 = max(0, min(y2, current_resolution[1]-1))
                # Extract mask in bbox area
                if hsv_mask is not None and hsv_mask.size > 0:
                    bbox_mask = hsv_mask[y1:y2, x1:x2]
                    if bbox_mask.size == 0:
                        continue
                    green_pixels = np.count_nonzero(bbox_mask)
                    #total_pixels = bbox_mask.size
                    #green_ratio = green_pixels / total_pixels
                    #if green_ratio > threshold:
                    if green_pixels:
                        boxes_inside_green.append((bbox, conf_score))
                        with open(log_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, conf_score, bbox])

        annotated_frame = resized_frame.copy()
        # Draw green boundary
        if hsv_mask is not None and hsv_mask.size > 0:
            contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_frame, contours, -1, (0,255,0), 2)
        # Draw boxes inside green
        for bbox, conf_score in boxes_inside_green:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(annotated_frame, f"{conf_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        if save_figs:
            output_file = os.path.join(output_dir, f'img_{timestamp}_{counter}_{int(person_detected)}.jpg')
            #output_file = os.path.join(output_dir, f'img_{timestamp}_{counter}.jpg')
            cv2.imwrite(output_file, cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        cv2.imshow("Camera", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = max(0, 1.0/fps - elapsed_time)
        time.sleep(sleep_time)
        counter = counter + 1

        if n_max > 0 and counter >= n_max:
            stop_flag = True
            break

        if mode_hsv:
            # shift to display_green
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) == ord("q"):
            stop_flag = True
            break

    cv2.destroyAllWindows()


def display_green():
    global stop_flag, mode_hsv, current_resolution, current_zoom, hsv_mask
    while not stop_flag and mode_hsv:

        start_time = time.time()

        frame = camera.capture_array()
        zoomed_frame = apply_zoom(frame, current_zoom)
        resized_frame = resize_frame(zoomed_frame, current_resolution)

        filtered_frame, hsv_frame, mask = filter_hsv_and_largest_component(resized_frame, h_low, s_low, v_low, h_high, s_high, v_high)
        hsv_mask = mask

        cv2.imshow('Filtered Preview', cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB))

        plot_filename = 'hsv_plot.png'
        save_hsv_plots(hsv_frame, plot_filename)
        plot_image = cv2.imread(plot_filename)
        plot_image_resized = resize_frame(plot_image, current_resolution)
        cv2.imshow('HSV Intensities', plot_image_resized)

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = max(0, 1.0/fps - elapsed_time)
        time.sleep(sleep_time)

        if not mode_hsv:
            # shift to display_persons
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            stop_flag = True
            break
        
    cv2.destroyAllWindows()


def get_user_input():
    global h_low, h_high, s_low, s_high, v_low, v_high, stop_flag, mode_hsv, current_resolution, current_zoom
    while not stop_flag:
        if mode_hsv:
            new_input = input("Enter HSV thresholds (e.g., h_low=0,h_high=255,s_low=0,s_high=255,v_low=0,v_high=255) or 'T' to switch modes or 'q' to quit: ")
            if new_input.lower() == 'q':
                stop_flag = True
                break
            elif new_input.lower() == 't':
                mode_hsv = not mode_hsv
                continue
            else:
                try:
                    for threshold in new_input.split(','):
                        key, value = threshold.split('=')
                        if key.strip() == 'h_low':
                            h_low = int(value)
                        elif key.strip() == 'h_high':
                            h_high = int(value)
                        elif key.strip() == 's_low':
                            s_low = int(value)
                        elif key.strip() == 's_high':
                            s_high = int(value)
                        elif key.strip() == 'v_low':
                            v_low = int(value)
                        elif key.strip() == 'v_high':
                            v_high = int(value)
                    print(f"Updated thresholds: h_low={h_low}, h_high={h_high}, s_low={s_low}, s_high={s_high}, v_low={v_low}, v_high={v_high}")
                except ValueError:
                    print("Invalid input format. Use format like 'h_low=0,h_high=179,s_low=0,s_high=255,v_low=0,v_high=255'.")
        
        else:
            new_input = input("Enter new resolution (e.g., 854x480), zoom (e.g., zoom 1.2), or 'T' to switch modes or 'q' to quit: ")
            if new_input.lower() == 'q':
                stop_flag = True
                break
            elif new_input.lower() == 't':
                mode_hsv = not mode_hsv
                continue
            elif new_input.lower().startswith('zoom'):
                try:
                    _, zoom_factor = new_input.split()
                    zoom_factor = float(zoom_factor)
                    if zoom_factor >= 1.0:
                        current_zoom = zoom_factor
                        print(f"Zoom updated to: {zoom_factor}x")
                    else:
                        print("Zoom factor must be 1.0 or greater.")
                except (ValueError, IndexError):
                    print("Invalid zoom format. Use format like 'zoom 1.2'.")
            else:
                try:
                    width, height = map(int, new_input.split('x'))
                    current_resolution = (width, height)
                    print(f"Resolution updated to: {width}x{height}")
                except ValueError:
                    print("Invalid resolution format. Use format like '854x480'.")


def run_display():
    global stop_flag, mode_hsv
    while not stop_flag:
        if mode_hsv:
            display_green()
        else:
            display_persons()

display_thread = threading.Thread(target=run_display)
input_thread = threading.Thread(target=get_user_input)


input_thread.start()
display_thread.start()

input_thread.join()
display_thread.join()

camera.close()
cv2.destroyAllWindows()