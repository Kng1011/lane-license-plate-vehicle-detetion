import cv2
import numpy as np
from lane_detection import detect_lanes, Lane
from vehicle_detection import VehicleDetector
import time
from tqdm import tqdm

def get_vehicle_lane_position(vehicle_center_x, left_lane_x, right_lane_x):
    """Determine which lane the vehicle is in"""
    lane_width = right_lane_x - left_lane_x
    if lane_width <= 0:
        return 'unknown'
    
    # Calculate relative position
    relative_pos = (vehicle_center_x - left_lane_x) / lane_width
    
    if relative_pos < 0.3:
        return 'left'
    elif relative_pos > 0.7:
        return 'right'
    else:
        return 'center'

def process_video(input_path, output_path, show_progress=True):
    """Process video with lane and vehicle detection"""
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize vehicle detector
    vehicle_detector = VehicleDetector()
    
    # Initialize lane objects
    left_lane = Lane()
    right_lane = Lane()
    
    # Initialize progress bar
    if show_progress:
        pbar = tqdm(total=total_frames, desc="Processing video")
    
    # Process frames
    frame_count = 0
    batch_frames = []
    batch_indices = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame to batch
        batch_frames.append(frame)
        batch_indices.append(frame_count)
        
        # Process batch when it reaches max size or at the end
        if len(batch_frames) >= vehicle_detector.max_batch_size or frame_count == total_frames - 1:
            # Process vehicle detection first
            vehicle_results, plate_results = vehicle_detector.process_batch(batch_frames)
            
            # Process each frame in the batch
            for i, (frame, vehicle_result, plate_result) in enumerate(zip(batch_frames, vehicle_results, plate_results)):
                if vehicle_result is None or plate_result is None:
                    continue
                
                # Detect lanes
                lane_frame, lane_info = detect_lanes(frame, left_lane, right_lane)
                
                # Draw vehicle and plate detections
                final_frame = vehicle_detector.draw_detections(lane_frame, [vehicle_result], [plate_result])
                
                # Add vehicle lane position information
                if left_lane.detected and right_lane.detected and left_lane.mean_fitx is not None and right_lane.mean_fitx is not None:
                    for result in vehicle_result:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls in vehicle_detector.VEHICLE_CLASS_ID:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                vehicle_center_x = (x1 + x2) / 2

                                # Get lane positions at vehicle's y position, ensuring y_pos is within bounds
                                y_pos = int(y2)
                                if 0 <= y_pos < len(left_lane.mean_fitx) and 0 <= y_pos < len(right_lane.mean_fitx):
                                    left_lane_x = left_lane.mean_fitx[y_pos]
                                    right_lane_x = right_lane.mean_fitx[y_pos]

                                    # Only calculate and display lane position if a valid lane width exists
                                    if right_lane_x > left_lane_x:
                                        lane_position = get_vehicle_lane_position(vehicle_center_x, left_lane_x, right_lane_x)

                                        # Add lane position to vehicle label
                                        class_name = vehicle_detector.vehicle_class_names[cls]
                                        color = vehicle_detector.VEHICLE_COLORS[cls]  # Corrected attribute name
                                        label = f"{class_name} ({lane_position})"
                                        cv2.putText(final_frame, label, (x1, y1-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Write frame
                out.write(final_frame)
                
                # Update progress
                if show_progress:
                    pbar.update(1)
            
            # Clear batch
            batch_frames = []
            batch_indices = []
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    if show_progress:
        pbar.close()
    
    print(f"\nProcessing complete. Output saved to {output_path}")

if __name__ == "__main__":
    # Process video
    input_video = "dash.mp4"  # Change this to your input video path
    output_video = "dash_output.mp4"  # Change this to your desired output path
    
    process_video(input_video, output_video) 