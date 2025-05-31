import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import torch

class VehicleDetector:
    # Define vehicle classes of interest as class attributes
    VEHICLE_CLASS_ID = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Define colors for vehicle classes (BGR format)
    VEHICLE_COLORS = {
        2: (0, 255, 0),    # Car - Green
        3: (255, 0, 0),    # Motorcycle - Blue
        5: (0, 0, 255),    # Bus - Red
        7: (255, 255, 0)   # Truck - Cyan
    }

    # Color for license plates
    PLATE_COLOR = (0, 165, 255)  # Orange

    def __init__(self, vehicle_model_path="yolov8x.pt", plate_model_path="best.pt"):
        # Initialize models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLO models
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        
        # Move models to GPU if available
        if self.device == 'cuda':
            self.vehicle_model.to(self.device)
            self.plate_model.to(self.device)
            
            # Set GPU properties
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            # Calculate optimal batch size
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            estimated_memory_per_frame = 300 * (1024**2)  # 300MB
            self.max_batch_size = min(int((gpu_memory * 0.4) / estimated_memory_per_frame), 4)
        else:
            self.max_batch_size = 1
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        
        # Get vehicle class names
        self.vehicle_class_names = self.vehicle_model.model.names
        
        # Create CUDA streams for parallel processing
        if self.device == 'cuda':
            self.vehicle_stream = torch.cuda.Stream()
            self.plate_stream = torch.cuda.Stream()

    def is_valid_plate_box(self, box, frame_shape, min_area=100, max_area=50000):
        """Check if a detected box is likely to be a vehicle license plate"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        relative_area = (area / frame_area) * 100
        
        is_valid = (
            min_area <= area <= max_area and
            2.0 <= aspect_ratio <= 5.0 and
            relative_area < 5.0 and
            height < frame_height * 0.15
        )
        
        return is_valid

    def process_batch(self, frames):
        """Process a batch of frames using GPU streams"""
        if self.device == 'cuda':
            try:
                # Maintain consistent GPU usage
                self.maintain_gpu_usage(0.4)
                
                # Process vehicle detection
                try:
                    with torch.cuda.stream(self.vehicle_stream):
                        vehicle_results = self.vehicle_model.predict(
                            frames, 
                            conf=0.2,
                            classes=self.VEHICLE_CLASS_ID,  # Use class attribute
                            device=self.device,
                            verbose=False
                        )
                except Exception as e:
                    print(f"Error in vehicle detection: {str(e)}")
                    return None, None
                
                # Small consistent delay
                import time
                time.sleep(0.05)
                
                # Process plate detection
                try:
                    with torch.cuda.stream(self.plate_stream):
                        plate_results = self.plate_model.predict(
                            frames, 
                            conf=0.15,
                            iou=0.3,
                            device=self.device,
                            verbose=False,
                            agnostic_nms=True,
                            imgsz=640,
                            augment=True
                        )
                except Exception as e:
                    print(f"Error in plate detection: {str(e)}")
                    return None, None
                
                # Synchronize streams
                torch.cuda.synchronize()
                
                # Maintain consistent GPU usage after processing
                self.maintain_gpu_usage(0.4)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\nGPU memory high, reducing batch size...")
                    torch.cuda.empty_cache()
                    time.sleep(0.5)
                    return self.process_batch(frames[:len(frames)//2])
                print(f"Runtime error in batch processing: {str(e)}")
                return None, None
            except Exception as e:
                print(f"Unexpected error in batch processing: {str(e)}")
                return None, None
        else:
            try:
                vehicle_results = self.vehicle_model.predict(frames, conf=0.2, classes=self.VEHICLE_CLASS_ID)  # Use class attribute
                plate_results = self.plate_model.predict(
                    frames, 
                    conf=0.15, 
                    iou=0.3, 
                    agnostic_nms=True,
                    imgsz=640,
                    augment=True
                )
            except Exception as e:
                print(f"Error in CPU processing: {str(e)}")
                return None, None
        
        return vehicle_results, plate_results

    def maintain_gpu_usage(self, target_usage=0.4):
        """Maintain consistent GPU usage"""
        if self.device == 'cuda':
            import time
            current_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if current_usage > target_usage:
                time.sleep(0.2)
                torch.cuda.empty_cache()
            elif current_usage < target_usage * 0.5:
                time.sleep(0.05)

    def draw_detections(self, frame, vehicle_results, plate_results):
        """Draw vehicle and license plate detections on frame"""
        frame_copy = frame.copy()
        height, width = frame_copy.shape[:2]
        
        # Draw vehicle detections
        for result in vehicle_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls not in self.VEHICLE_CLASS_ID:  # Use class attribute
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                class_name = self.vehicle_class_names[cls]
                color = self.VEHICLE_COLORS[cls]  # Use class attribute
                
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame_copy, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw license plate detections
        for result in plate_results:
            boxes = result.boxes
            for box in boxes:
                if not self.is_valid_plate_box(box, frame_copy.shape):
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < 0.15:
                    continue
                    
                padding = 5
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(width, x2 + padding)
                y2_pad = min(height, y2 + padding)
                
                plate_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if plate_region.size > 0:
                    plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    plate_gray = cv2.GaussianBlur(plate_gray, (3, 3), 0)
                    plate_gray = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY, 11, 2)
                    
                    ocr_results = self.reader.readtext(plate_region)
                    if not ocr_results:
                        ocr_results = self.reader.readtext(plate_gray)
                    
                    if ocr_results:
                        text = ocr_results[0][1]
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), self.PLATE_COLOR, 2)  # Use class attribute
                        cv2.putText(frame_copy, f"Plate: {text} ({conf:.2f})", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.PLATE_COLOR, 2)  # Use class attribute
                    else:
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), self.PLATE_COLOR, 2)  # Use class attribute
                        cv2.putText(frame_copy, f"Plate ({conf:.2f})", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.PLATE_COLOR, 2)  # Use class attribute
        
        return frame_copy 