from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from tqdm import tqdm
import time

# Load YOLO models
vehicle_model_path = "yolov8x.pt"  # Model for vehicle detection
plate_model_path = "best.pt"  # Model for license plate detection

# Initialize models
vehicle_model = YOLO(vehicle_model_path)
plate_model = YOLO(plate_model_path)

# Define vehicle classes of interest
VEHICLE_CLASS_ID = [2, 3, 5, 7]  # car, motorcycle, bus, truck
vehicle_class_names = vehicle_model.model.names

# Define colors for vehicle classes (BGR format)
VEHICLE_COLORS = {
    2: (0, 255, 0),    # Car - Green
    3: (255, 0, 0),    # Motorcycle - Blue
    5: (0, 0, 255),    # Bus - Red
    7: (255, 255, 0)   # Truck - Cyan
}

# Color for license plates
PLATE_COLOR = (0, 165, 255)  # Orange

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Lane detection parameters
LANEWIDTH = 3.7  # highway lane width in meters
N = 4  # number of frames to average over

# Perspective transform points
frame_width = 1280
frame_height = 720
input_scale = 1

# Source points for perspective transform
x = [194, 1117, 705, 575]
y = [719, 719, 461, 461]
X = [290, 990, 990, 290]
Y = [719, 719, 0, 0]

src = np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]]) / input_scale
dst = np.float32([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]]) / input_scale

# Calculate perspective transform matrices
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

class Lane:
    def __init__(self):
        self.detected = False
        self.cur_fitx = None
        self.cur_fity = None
        self.prev_fitx = []
        self.current_poly = [np.array([False])]
        self.prev_poly = [np.array([False])]
        self.mean_fitx = None

    def average_pre_lanes(self):
        if self.cur_fitx is not None:
            tmp = self.prev_fitx.copy()
            tmp.append(self.cur_fitx)
            self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == N:
            self.prev_fitx.pop(0)
        if self.mean_fitx is not None:
            self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly

# Initialize lane objects
left_lane = Lane()
right_lane = Lane()

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def find_edges(img, s_thresh=(120, 255), sx_thresh=(20, 100), dir_thresh=(0.7, 1.3)):
    img = np.copy(img)
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float64)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # Gradient direction
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)

    # Combine thresholds
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((sxbinary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (dir_binary == 1))] = 1

    # Add more weight to s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[(sxbinary == 1) & (s_binary == 1)] = 2

    return (combined_binary + c_bi)

def full_search(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int64(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    window_height = np.int64(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Check if we found enough points
    if len(left_lane_inds) < 100 or len(right_lane_inds) < 100:
        # Return None to indicate detection failure
        return None, None

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit
    except:
        return None, None

def measure_lane_curvature(ploty, leftx, rightx):
    # Convert from pixels to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = LANEWIDTH/700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Determine curve direction - Inverted logic
    if leftx[-1] - leftx[0] > 50:  # Changed from leftx[0] - leftx[-1]
        curve_direction = 'left curve'
    elif leftx[0] - leftx[-1] > 50:  # Changed from leftx[-1] - leftx[0]
        curve_direction = 'right curve'
    else:
        curve_direction = 'straight'

    return (left_curverad + right_curverad)/2.0, curve_direction

def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):
    # Calculate the offset from center
    height = undist.shape[0]
    width = undist.shape[1]
    
    # Get the bottom points of the lanes
    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]
    
    # Calculate the center of the lane
    lane_center = (bottom_l + bottom_r) / 2
    # Calculate the offset from image center
    image_center = width / 2
    offset = (lane_center - image_center) * LANEWIDTH / (bottom_r - bottom_l)
    
    # Determine deviation direction
    if abs(offset) < 0.1:
        dev_dir = 'center'
    else:
        dev_dir = 'left' if offset < 0 else 'right'
    
    return offset, dev_dir

def detect_lanes(frame):
    """Lane detection using perspective transform and sliding windows"""
    # Convert to grayscale and find edges
    img_binary = find_edges(frame)
    
    # Apply perspective transform
    binary_warped = cv2.warpPerspective(img_binary, M, (frame_width, frame_height))
    
    # Crop the binary image
    binary_sub = np.zeros_like(binary_warped)
    binary_sub[:, 150:-80] = binary_warped[:, 150:-80]
    
    # Generate points for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Detect lanes
    if left_lane.detected:
        # Use previous detection to search around
        left_fit, right_fit = full_search(binary_sub)
    else:
        # Full search
        left_fit, right_fit = full_search(binary_sub)
    
    # If detection failed, use previous detection or return default values
    if left_fit is None or right_fit is None:
        if len(left_lane.prev_fitx) > 0:
            # Use previous detection
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
            left_lane.detected = False
            right_lane.detected = False
        else:
            # Return default straight lane
            lane_info = {
                'left_curverad': 0,
                'right_curverad': 0,
                'center_dist': 0,
                'curve_direction': 'straight',
                'curvature': 0,
                'dev_dir': 'center',
                'offset': 0.0
            }
            return frame.copy(), lane_info
    else:
        # Generate x values for plotting
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Check if detection is valid
        std_value = np.std(right_fitx - left_fitx)
        if std_value < 85:
            left_lane.detected = True
            right_lane.detected = True
            left_lane.current_poly = left_fit
            right_lane.current_poly = right_fit
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        else:
            left_lane.detected = False
            right_lane.detected = False
            if len(left_lane.prev_fitx) > 0:
                left_lane.cur_fitx = left_lane.prev_fitx[-1]
                right_lane.cur_fitx = right_lane.prev_fitx[-1]
            else:
                left_lane.cur_fitx = left_fitx
                right_lane.cur_fitx = right_fitx
    
    # Process lanes
    left_lane.process(ploty)
    right_lane.process(ploty)
    
    # Measure curvature and direction
    curvature, curve_direction = measure_lane_curvature(ploty, left_lane.mean_fitx, right_lane.mean_fitx)
    
    # Compute offset
    offset, dev_dir = compute_car_offcenter(ploty, left_lane.mean_fitx, right_lane.mean_fitx, frame)
    
    # Create lane info dictionary
    lane_info = {
        'left_curverad': curvature,
        'right_curverad': curvature,
        'center_dist': 0,
        'curve_direction': curve_direction,
        'curvature': curvature,
        'dev_dir': dev_dir,
        'offset': offset
    }
    
    # Draw lanes on frame
    lane_frame = frame.copy()
    pts_left = np.array([np.transpose(np.vstack([left_lane.mean_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.mean_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane
    cv2.fillPoly(lane_frame, np.int_([pts]), (0, 255, 0))
    
    return lane_frame, lane_info

def draw_detections(frame, vehicle_results, plate_results, lane_info):
    """Draw vehicle detections, license plates, and lane information on frame"""
    frame_copy = frame.copy()
    
    # Draw vehicle detections
    for result in vehicle_results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASS_ID:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            class_name = vehicle_class_names[cls]
            color = VEHICLE_COLORS[cls]
            
            # Draw vehicle rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Add class name and confidence
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Draw license plate detections
    for result in plate_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop plate region
            plate_region = frame[y1:y2, x1:x2]
            
            # Recognize plate text
            if plate_region.size > 0:
                ocr_results = reader.readtext(plate_region)
                if ocr_results:
                    text = ocr_results[0][1]
                    cv2.putText(frame_copy, f"Plate: {text}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, PLATE_COLOR, 2)
            
            # Draw plate rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), PLATE_COLOR, 2)
    
    # Add lane information with more details
    cv2.putText(frame_copy, f"Lane: {lane_info['curve_direction']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Curvature: {lane_info['curvature']:.1f}m", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Offset: {lane_info['offset']:.2f}m ({lane_info['dev_dir']})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame_copy

def process_video(video_path, output_path=None):
    """Process video with vehicle detection, license plate detection, and lane detection"""
    # Get video information
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Setup video writer if needed
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Vehicle detection
        vehicle_results = vehicle_model.predict(frame, conf=0.25, classes=VEHICLE_CLASS_ID)
        
        # License plate detection
        plate_results = plate_model.predict(frame, conf=0.25)
        
        # Lane detection
        lane_frame, lane_info = detect_lanes(frame)
        
        # Draw all detections
        frame_with_detections = draw_detections(frame, vehicle_results, plate_results, lane_info)
        
        # Save frame if needed
        if writer:
            writer.write(frame_with_detections)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    print("Processing complete!")

if __name__ == "__main__":
    input_video = "project_video.mp4" 
    output_video = "output_combined2.mp4" 
    
    process_video(input_video, output_video) 