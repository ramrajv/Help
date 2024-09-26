### 1. **Setup: Install Required Libraries**
You will need the following libraries. Install them via `pip`:
```bash
pip install opencv-python torch torchvision numpy sort
```

### 2. **Object Detection (YOLOv5)**
Let's start by using a pre-trained YOLO model for detecting objects in both the EO and IR video streams.

#### YOLO Setup for EO and IR
We'll use YOLOv5 from PyTorch Hub:

```python
import cv2
import torch

# Load pre-trained YOLOv5 model from PyTorch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s is fast and light
model.eval()

# Initialize video streams (EO and IR)
eo_video = cv2.VideoCapture('eo_video.mp4')
ir_video = cv2.VideoCapture('ir_video.mp4')

def detect_objects(frame):
    # Use YOLOv5 for object detection
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Extract detections (x1, y1, x2, y2, confidence, class)
    return detections
```

### 3. **Object Tracking (SORT)**
Next, we'll implement tracking using the SORT algorithm, which is lightweight and easy to integrate.

Install SORT with:
```bash
pip install filterpy
```

Now, define the SORT tracker and integrate it with the object detection pipeline:

```python
from sort import Sort

# Initialize SORT tracker
tracker = Sort()

def track_objects(detections):
    # Convert detections to [x1, y1, x2, y2, score] format for SORT
    sort_input = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        sort_input.append([x1, y1, x2, y2, conf])
    
    # Update the tracker
    tracked_objects = tracker.update(np.array(sort_input))
    
    return tracked_objects
```

### 4. **Fusion of EO and IR Detections**
We need to align detections from both streams by using simple feature matching or temporal consistency.

```python
import numpy as np

def fuse_detections(eo_detections, ir_detections):
    # Simple example of fusing detections based on overlapping bounding boxes.
    fused_detections = []
    for eo_det in eo_detections:
        for ir_det in ir_detections:
            if iou(eo_det, ir_det) > 0.5:  # Fuse based on Intersection-over-Union (IoU)
                fused_detections.append(eo_det)  # Or ir_det, depending on priority
    return fused_detections

def iou(box1, box2):
    # Compute Intersection over Union (IoU) for two bounding boxes
    x1, y1, x2, y2 = box1[:4]
    x1b, y1b, x2b, y2b = box2[:4]
    
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    
    iou_value = inter_area / (box1_area + box2_area - inter_area)
    return iou_value
```

### 5. **Main Loop: Video Capture and Processing**
Here is how you would process the video streams, detect objects, fuse them, and then track them in real-time:

```python
while eo_video.isOpened() and ir_video.isOpened():
    # Read frames from EO and IR video streams
    ret_eo, eo_frame = eo_video.read()
    ret_ir, ir_frame = ir_video.read()
    
    if not ret_eo or not ret_ir:
        break
    
    # Detect objects in both EO and IR frames
    eo_detections = detect_objects(eo_frame)
    ir_detections = detect_objects(ir_frame)
    
    # Fuse EO and IR detections
    fused_detections = fuse_detections(eo_detections, ir_detections)
    
    # Track fused objects
    tracked_objects = track_objects(fused_detections)
    
    # Draw tracking results on the EO frame
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(eo_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(eo_frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the EO frame with tracking results
    cv2.imshow('EO Video with Tracking', eo_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
eo_video.release()
ir_video.release()
cv2.destroyAllWindows()
```

### 6. **Additional Considerations:**
- **Real-Time Performance**: For real-time processing, make sure to optimize the detection and tracking loops using hardware acceleration like CUDA or TensorRT for deep learning models.
- **Rotating Camera**: If the camera rotation is affecting tracking, implement an additional stabilization algorithm using techniques like optical flow or camera pose estimation to handle background changes.

This example should give you a starting point. If you want to explore any section in detail (e.g., fine-tuning models, handling more complex fusion), let me know!
