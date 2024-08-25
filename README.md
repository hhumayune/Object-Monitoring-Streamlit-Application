# Object-Monitoring-YOLOv8-StreamLit
An object detection and tracking application made on StreamLit that uses YOLOv8 models.

![image](https://github.com/user-attachments/assets/2d212d83-c422-4857-aa2f-693c6a3c1bfa)
## Features

### 1. Input Options

- **Images and Videos**: Upload images or video files directly from your device.
- **Webcam Feeds**: Use your computer's webcam to analyze real-time video feeds.
- **YouTube Links**: Analyze video content directly from YouTube by providing a URL.

  
### 2. Model Selection

You can chose any YOLOv8 model by uploading it to the 'models' folder.

### 3. Hardware Configuration

Toggle between CPU and GPU (CUDA) modes depending on your available hardware resources. The application automatically shifts to CPU if no GPU is detected.

### 4. Tracking Algorithms
You can select from BoTSORT and ByteTrack.

### 5. Confidence Threshold Control
You can adjust the model’s confidence threshold using a slider. This allows users to fine-tune detection sensitivity to meet specific operational requirements.

### 6. FPS Tracker
An FPS counter is also provided to check performance.

### 7. Results
User has the ability to save the results file after processing has finished.
