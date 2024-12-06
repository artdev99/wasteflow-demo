import time
import os
import cv2
import shutil
from picamera2 import CameraConfiguration
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt 
from scipy.ndimage import label

#RESOLUTION = (640, 480) # 854, 480
FPS = 0.25 # 4 secondes 
N_MAX = 28800
ZOOM = 1.0
PERSON_LABEL = 0 # YOLO, COCO dataset label
WINDOW_SIZE = 11
BITRATE = 8
CONFIDENCE_THRESHOLD = 0.5

############## CAMERA ##############
def start_camera(camera, resolution=(640,480), format="BGR888", raw=False, raw_format="SRGGB8", verbose=False):
    """
    Configures and starts the camera for still image capture.
    """
    camera_config_dict = camera.create_still_configuration()
    camera_config = CameraConfiguration(camera_config_dict, picam2=camera)

    if verbose:
        print(type(camera_config_dict), type(camera_config))
    
    if raw:
        camera_config.enable_raw(True)
        camera_config.raw.size = resolution
        camera_config.raw.format = raw_format
        if verbose:
            print("Enabled raw stream")
            print(f"Raw size: {resolution} format: {raw_format}")
    else:
        camera_config.enable_raw(False)
        if verbose:
            print("Disabled raw stream")
    
    camera_config.main.size = resolution
    camera_config.main.format = format
    camera_config.sensor.bit_depth = 8 # force SBGGR8 on sensor

    if verbose:
        print(f"Main size: {resolution} format: {format}")

    camera_config.align()

    if verbose:
        print(camera_config)

    camera.configure(camera_config)
    camera.start()
    time.sleep(1)

def resize_frame(frame, resolution):
    """Resize the frame to the desired resolution."""
    if len(frame.shape) > 2:
        h,w,_ = frame.shape    
    else:
        h,w = frame.shape # grayscale image 
    fs = (w, h)
    if fs == resolution:
        return frame
    else:
        return cv2.resize(frame, resolution)

def apply_zoom_2(frame, zoom_factor):
    """Apply digital zoom around the center of the frame."""
    if zoom_factor <= 1.0:
        return frame

    # Calculate the center crop dimensions based on the zoom factor
    height, width = frame.shape[:2]
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Calculate cropping coordinates to keep the crop centered
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    # Crop the frame to the new dimensions
    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]

    return cropped_frame

def apply_zoom(frame, zoom_factor):
    """Modifies the shape of the frame"""
    if zoom_factor <= 1.0:
            return frame

    h, w = frame.shape[:2]
    # Calculate the center of the image
    center_x, center_y = w / 2, h / 2
    # Calculate the new boundaries
    radius_x, radius_y = w / (2 * zoom_factor), h / (2 * zoom_factor)
    min_x, max_x = int(center_x - radius_x), int(center_x + radius_x)
    min_y, max_y = int(center_y - radius_y), int(center_y + radius_y)
    # Ensure the boundaries are within the image
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)
    # Crop and return the zoomed image
    cropped = frame[min_y:max_y, min_x:max_x]
    return cropped

############## FOLDER ##############
def create_folder_delete_if_exists(output_dir, require_user=False):
    if os.path.exists(output_dir):
        if require_user:
            user_input = input(f"The folder '{output_dir}' already exists. Do you want to delete it and recreate it? (yes/no): ").strip().lower()
            if user_input == 'yes':
                shutil.rmtree(output_dir)
                print(f"Folder '{output_dir}' has been deleted.")
            else:
                print("Exiting program. Folder not deleted.")
                exit()
        else:
            shutil.rmtree(output_dir)
            print(f"Folder '{output_dir}' has been deleted.")
    
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' has been created.")



############## DETECTION ##############
def smooth_detections(detection_list, window_size=WINDOW_SIZE):
    """
    Performs majority vote over a list of 11 detection results.
    Returns True if the majority are True, else False.
    """
    assert window_size % 2 == 1, "Window size must be odd"
    assert len(detection_list) == window_size, "Detection list must be of length window_size"
    true_count = sum(detection_list)
    return true_count >= window_size//2 + 1

def detect_person(model, img, confidence=CONFIDENCE_THRESHOLD):
    result = model.predict(img, classes=[PERSON_LABEL], verbose=False, conf=confidence)[0]
    if result.boxes and PERSON_LABEL in result.boxes.cls.int().tolist():
        return True
    else:
        return False
    

############## HSV ##############

def save_hsv_plots(hsv_frame, filename):
    """Plot and save HSV intensity along the central x and y lines using matplotlib."""
    center_x, center_y = hsv_frame.shape[1] // 2, hsv_frame.shape[0] // 2
    h_line_x = hsv_frame[center_y, :, 0]
    s_line_x = hsv_frame[center_y, :, 1]
    v_line_x = hsv_frame[center_y, :, 2]

    h_line_y = hsv_frame[:, center_x, 0]
    s_line_y = hsv_frame[:, center_x, 1]
    v_line_y = hsv_frame[:, center_x, 2]

    plt.figure(figsize=(12, 6))

    # X-axis plots
    plt.subplot(2, 3, 1)
    plt.plot(h_line_x, color='r')
    plt.title('Hue along X-axis')

    plt.subplot(2, 3, 2)
    plt.plot(s_line_x, color='g')
    plt.title('Saturation along X-axis')

    plt.subplot(2, 3, 3)
    plt.plot(v_line_x, color='b')
    plt.title('Value along X-axis')

    # Y-axis plots
    plt.subplot(2, 3, 4)
    plt.plot(h_line_y, color='r')
    plt.title('Hue along Y-axis')

    plt.subplot(2, 3, 5)
    plt.plot(s_line_y, color='g')
    plt.title('Saturation along Y-axis')

    plt.subplot(2, 3, 6)
    plt.plot(v_line_y, color='b')
    plt.title('Value along Y-axis')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def filter_hsv_and_largest_component(frame, h_low, s_low, v_low, h_high, s_high, v_high):
    """Filter the frame using HSV thresholding and keep only the strongest component."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, (h_low, s_low, v_low), (h_high, s_high, v_high))

    # largest connected component
    labeled, num_features = label(mask)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0 # skip background
        #largest_label = sizes[1:].argmax() + 1
        largest_label = sizes.argmax()
        mask = np.where(labeled == largest_label, 255, 0).astype(np.uint8)
    else:
        mask = np.zeros_like(mask)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, hsv_frame, mask