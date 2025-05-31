import cv2
import numpy as np

# Lane detection parameters
LANEWIDTH = 3.7  # highway lane width in meters
N = 4  # number of frames to average over

class Lane:
    def __init__(self):
        self.detected = False
        self.cur_fitx = None
        self.cur_fity = None
        self.prev_fitx = []
        self.current_poly = [np.array([False])]
        self.prev_poly = [np.array([False])]
        self.mean_fitx = None
        self.detection_failures = 0
        self.max_failures = 5
        self.curve_direction = 'straight'
        self.curvature = 0
        self.offset = 0
        self.dev_dir = 'center'

    def average_pre_lanes(self):
        if self.cur_fitx is not None:
            tmp = self.prev_fitx.copy()
            tmp.append(self.cur_fitx)
            if len(tmp) > 0:
                self.mean_fitx = np.mean(tmp, axis=0)
                if len(self.prev_fitx) > 0:
                    self.mean_fitx = 0.7 * self.mean_fitx + 0.3 * self.prev_fitx[-1]

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

def get_perspective_points(frame_width, frame_height):
    """Calculate perspective transform points for current frame size"""
    x_ratio = frame_width / 1280.0
    y_ratio = frame_height / 720.0
    
    x = [194, 1117, 705, 575]
    y = [719, 719, 461, 461]
    X = [290, 990, 990, 290]
    Y = [719, 719, 0, 0]
    
    x = [int(xi * x_ratio) for xi in x]
    y = [int(yi * y_ratio) for yi in y]
    X = [int(Xi * x_ratio) for Xi in X]
    Y = [int(Yi * y_ratio) for Yi in Y]
    
    src = np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]])
    dst = np.float32([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]])
    
    return src, dst

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Apply Sobel thresholding"""
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
    """Apply direction thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def find_edges(img, s_thresh=(100, 255), sx_thresh=(20, 100), dir_thresh=(0.7, 1.3)):
    """Find edges in the image with improved parameters"""
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float64)
    s_channel = hls[:,:,2]
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    s_channel = clahe.apply(np.uint8(s_channel))
    
    # Normalize the S channel
    s_channel = cv2.normalize(s_channel, None, 0, 255, cv2.NORM_MINMAX)
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=5, thresh=sx_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=5, thresh=dir_thresh)

    combined_binary = np.zeros_like(s_channel)
    combined_binary[((sxbinary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    c_bi = np.zeros_like(s_channel)
    c_bi[(sxbinary == 1) & (s_binary == 1)] = 2

    kernel = np.ones((5,5), np.uint8)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel)

    return (combined_binary + c_bi)

def full_search(binary_warped):
    """Perform full search for lane detection"""
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int64(binary_warped.shape[0]/nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100
    minpix = 30
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    if len(left_lane_inds) < 100 or len(right_lane_inds) < 100:
        return None, None

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit
    except:
        return None, None

def measure_lane_curvature(ploty, leftx, rightx):
    """Measure lane curvature with improved accuracy"""
    ym_per_pix = 30/720
    xm_per_pix = LANEWIDTH/700

    centerx = (leftx + rightx) / 2
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    center_fit_cr = np.polyfit(ploty*ym_per_pix, centerx*xm_per_pix, 2)
    
    y_eval = np.max(ploty)
    center_curverad = ((1 + (2*center_fit_cr[0]*y_eval*ym_per_pix + center_fit_cr[1])**2)**1.5) / np.absolute(2*center_fit_cr[0])
    
    center_derivative = 2*center_fit_cr[0]*y_eval*ym_per_pix + center_fit_cr[1]
    curve_angle = np.arctan(center_derivative) * 180 / np.pi
    
    if center_curverad < 1000:  # Sharp curve
        if curve_angle > 5:
            curve_direction = 'right curve'
        elif curve_angle < -5:
            curve_direction = 'left curve'
        else:
            curve_direction = 'straight'
    else:  # Gentle curve
        if curve_angle > 2:
            curve_direction = 'right curve'
        elif curve_angle < -2:
            curve_direction = 'left curve'
        else:
            curve_direction = 'straight'
    
    return center_curverad, curve_direction

def compute_car_offcenter(ploty, left_fitx, right_fitx, frame_width):
    """Calculate the offset from center with improved accuracy"""
    # Get the bottom points of the lanes
    bottom_l = left_fitx[-1]
    bottom_r = right_fitx[-1]
    
    # Calculate lane width and center
    lane_width = bottom_r - bottom_l
    lane_center = (bottom_l + bottom_r) / 2
    image_center = frame_width / 2
    
    # Calculate offset in meters
    xm_per_pix = LANEWIDTH / 700  # meters per pixel in x dimension
    offset = (lane_center - image_center) * xm_per_pix
    
    # Determine deviation direction with a small deadzone
    if abs(offset) < 0.1:
        dev_dir = 'center'
    elif offset < 0:
        dev_dir = 'left'
    else:
        dev_dir = 'right'
    
    return offset, dev_dir

def detect_lanes(frame, left_lane, right_lane):
    """Main lane detection function with improved accuracy"""
    frame_height, frame_width = frame.shape[:2]
    
    src, dst = get_perspective_points(frame_width, frame_height)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # Enhanced edge detection
    img_binary = find_edges(frame)
    binary_warped = cv2.warpPerspective(img_binary, M, (frame_width, frame_height))
    
    # Additional processing for better lane detection
    kernel = np.ones((5,5), np.uint8)
    binary_warped = cv2.morphologyEx(binary_warped, cv2.MORPH_CLOSE, kernel)
    binary_warped = cv2.morphologyEx(binary_warped, cv2.MORPH_OPEN, kernel)
    
    # Adjust crop margins for better lane detection
    binary_sub = np.zeros_like(binary_warped)
    crop_margin = int(frame_width * 0.10)
    binary_sub[:, crop_margin:-crop_margin] = binary_warped[:, crop_margin:-crop_margin]
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Lane detection with improved validation
    if left_lane.detected and right_lane.detected and left_lane.detection_failures < left_lane.max_failures:
        left_fit, right_fit = full_search(binary_sub)
    else:
        left_fit, right_fit = full_search(binary_sub)
    
    if left_fit is None or right_fit is None:
        left_lane.detection_failures += 1
        right_lane.detection_failures += 1
        
        if len(left_lane.prev_fitx) > 0 and len(right_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
            left_lane.detected = True
            right_lane.detected = True
        else:
            if left_lane.detection_failures >= left_lane.max_failures:
                left_lane.detection_failures = 0
                right_lane.detection_failures = 0
                left_lane.detected = False
                right_lane.detected = False
                left_lane.prev_fitx = []
                right_lane.prev_fitx = []
            
            return frame.copy(), None
    else:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Calculate center line
        center_fitx = (left_fitx + right_fitx) / 2
        
        # Improved validation metrics
        std_value = np.std(right_fitx - left_fitx)
        lane_width = np.mean(right_fitx - left_fitx)
        min_lane_width = frame_width * 0.10
        max_lane_width = frame_width * 0.40
        
        # Calculate curvature and curve direction
        curvature, curve_direction = measure_lane_curvature(ploty, left_fitx, right_fitx)
        
        # Calculate offset
        offset, dev_dir = compute_car_offcenter(ploty, left_fitx, right_fitx, frame_width)
        
        # More lenient validation with additional checks
        if (std_value < 150 and std_value > 5 and
            min_lane_width < lane_width < max_lane_width and
            np.all(np.diff(left_fitx) > -50) and
            np.all(np.diff(right_fitx) < 50)):
            
            left_lane.detection_failures = 0
            right_lane.detection_failures = 0
            
            left_lane.detected = True
            right_lane.detected = True
            left_lane.current_poly = left_fit
            right_lane.current_poly = right_fit
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
            
            # Store additional information
            left_lane.curve_direction = curve_direction
            right_lane.curve_direction = curve_direction
            left_lane.curvature = curvature
            right_lane.curvature = curvature
            left_lane.offset = offset
            right_lane.offset = offset
            left_lane.dev_dir = dev_dir
            right_lane.dev_dir = dev_dir
        else:
            left_lane.detection_failures += 1
            right_lane.detection_failures += 1
            
            if len(left_lane.prev_fitx) > 0 and len(right_lane.prev_fitx) > 0:
                # Increased smoothing for more stable detection
                left_lane.cur_fitx = 0.9 * left_lane.prev_fitx[-1] + 0.1 * left_fitx
                right_lane.cur_fitx = 0.9 * right_lane.prev_fitx[-1] + 0.1 * right_fitx
                left_lane.detected = True
                right_lane.detected = True
            else:
                left_lane.cur_fitx = left_fitx
                right_lane.cur_fitx = right_fitx
    
    left_lane.process(ploty)
    right_lane.process(ploty)
    
    # Create output frame with improved visualization
    lane_frame = frame.copy()
    
    # Add information text - only draw if lanes are detected and meet criteria
    if left_lane.detected and right_lane.detected and left_lane.mean_fitx is not None and right_lane.mean_fitx is not None:
        info_text = []
        if hasattr(left_lane, 'curve_direction'):
            info_text.append(f"Curve: {left_lane.curve_direction}")
        if hasattr(left_lane, 'curvature'):
            info_text.append(f"Curvature: {int(left_lane.curvature)}m")
        if hasattr(left_lane, 'offset'):
            info_text.append(f"Offset: {left_lane.offset:.2f}m {left_lane.dev_dir}")
        
        # Display information with improved formatting
        y_pos = 30
        for text in info_text:
            cv2.putText(lane_frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 30
    
    return lane_frame, {
        'curve_direction': left_lane.curve_direction if hasattr(left_lane, 'curve_direction') else None,
        'curvature': left_lane.curvature if hasattr(left_lane, 'curvature') else None,
        'offset': left_lane.offset if hasattr(left_lane, 'offset') else None,
        'dev_dir': left_lane.dev_dir if hasattr(left_lane, 'dev_dir') else None
    } 