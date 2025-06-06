# Lane, License Plate, and Vehicle Detection System

## Authors

| Name | Student Number |
|------|----------------|
| José  Cruz | EI29436 |
| Joel Amaral | EI29369 |

---

This project implements a comprehensive computer vision system that combines lane detection, vehicle detection, and license plate recognition capabilities. It's designed to process video streams (such as dashcam footage) and provide real-time analysis of road conditions, vehicles, and their license plates.

## Features

### Lane Detection and Analysis
- **Real-time Lane Boundary Detection**
  - Uses computer vision techniques to identify lane markings
  - Implements perspective transformation to get a bird's-eye view of the road
  - Processes both straight and curved road segments

- **Lane Measurements**
  - **Curvature Calculation**: Measures the radius of the road curve in meters
    - Helps determine how sharp the upcoming turn is
    - Uses polynomial fitting to calculate the curve's radius
    - Converts pixel measurements to real-world distances
  
  - **Vehicle Position (Offset)**
    - Calculates how far the vehicle is from the center of the lane
    - Positive offset means the vehicle is to the right of center
    - Negative offset means the vehicle is to the left of center
    - Helps in lane departure warning systems

  - **Lane Width**
    - Measures the width of the current lane
    - Helps in determining if the vehicle is in a standard lane
    - Useful for detecting lane changes or merging situations

### Vehicle Detection
- **Multi-class Vehicle Recognition**
  - Detects and classifies different types of vehicles:
    - Cars (Class ID: 2)
    - Motorcycles (Class ID: 3)
    - Buses (Class ID: 5)
    - Trucks (Class ID: 7)
  - Each vehicle type is color-coded for easy identification:
    - Cars: Green
    - Motorcycles: Blue
    - Buses: Red
    - Trucks: Cyan

- **Detection Features**
  - Real-time bounding box detection with confidence scores
  - Tracks vehicles across frames
  - Handles multiple vehicles simultaneously
  - Works in various lighting conditions

### License Plate Recognition
- **Plate Detection and Reading**
  - Automatically locates license plates on detected vehicles
  - Uses OCR (Optical Character Recognition) to read plate numbers
  - Validates detected plates based on:
    - Size and aspect ratio
    - Position relative to the vehicle
    - Minimum area requirements
  - Real-time display of recognized plate numbers

## Technical Implementation

### Lane Detection Process
1. **Image Preprocessing**
   - Converts to grayscale
   - Applies Gaussian blur to reduce noise
   - Uses Sobel operators for edge detection

2. **Perspective Transform**
   - Converts the road view to a bird's-eye perspective
   - Enables accurate measurement of:
     - Lane curvature
     - Vehicle position
     - Road width

3. **Lane Line Detection**
   - Uses sliding window technique to find lane pixels
   - Fits polynomial curves to lane markings
   - Averages measurements over multiple frames for stability

### Vehicle and Plate Detection
1. **YOLO-based Detection**
   - Uses YOLOv8x for vehicle detection
   - Custom YOLO model for license plate detection
   - GPU-accelerated processing for real-time performance

2. **OCR Processing**
   - EasyOCR for text recognition
   - Post-processing to improve accuracy
   - Validation of detected text

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended for optimal performance)
- Required Python packages:
  - OpenCV (cv2)
  - NumPy
  - PyTorch
  - Ultralytics YOLO
  - EasyOCR
  - tqdm

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd lane-license-plate-vehicle-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required model weights:
   - YOLOv8x model for vehicle detection (`yolov8x.pt`)
   - Custom model for license plate detection (`best.pt`)

## Usage

### Running the Project

To run the complete system with all features:

```bash
python main.py
```

The main script (`main.py`) integrates both lane detection and vehicle/plate detection capabilities by utilizing:
- `lane_detection.py` for lane detection and analysis
- `vehicle_detection.py` for vehicle and license plate detection

## Performance Optimization

The system includes several optimizations:
- GPU acceleration with CUDA support
- Batch processing for improved throughput
- Memory management for stable GPU usage
- Parallel processing using CUDA streams

## Output

The system generates:
- Processed video with visual overlays showing:
  - Detected lanes with curvature measurements
  - Vehicle bounding boxes with type labels and confidence scores
  - License plate detections with recognized text
  - Vehicle position relative to lane center
  - Real-time measurements of:
    - Lane curvature (in meters)
    - Vehicle offset from center
    - Lane width

## Acknowledgments

This project was developed using concepts and implementations from the following repositories as references:

1. [License Plate Detection and Recognition using YOLOv8](https://github.com/Ammar-Abdelhady-ai/Licence-Plate-Detection-and-Recognition-using-YOLO-V8.git)
   - Used for license plate detection and recognition implementation
   - Referenced for YOLO model integration and OCR processing

2. [Vehicle Detection](https://github.com/JunshengFu/vehicle-detection.git)
   - Used for vehicle detection pipeline implementation
   - Referenced for YOLO-based detection approach and optimization techniques
