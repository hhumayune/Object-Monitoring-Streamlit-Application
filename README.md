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

## Installation

### Prerequisites
- Python 3.12 installed
- Terminal or command prompt access

### Setup Virtual Environment

1. Clone the repository or download the project files.

2. Open a terminal or command prompt.

3. Navigate to the project directory.

### For Windows:

#### Create a virtual environment
python -m venv venv

#### Activate the virtual environment
venv\Scripts\activate

#### Install dependencies
pip install -r requirements.txt


### For macOS/Linux:


#### Create a virtual environment
python3 -m venv venv

#### Activate the virtual environment
source venv/bin/activate

#### Install dependencies
pip install -r requirements.txt


### Running the Application

1. Ensure your virtual environment is activated.

2. In the terminal or command prompt, run the Streamlit application using the following command:

streamlit run main.py


3. This command will open a new tab in your default web browser with the Streamlit application running.

### Using the Application

- Select the input type (Image, Video, Webcam) and configure other parameters (Model, Hardware, Tracker, Model Confidence) using the sidebar.
- Upload an image or video file, or select webcam mode to start processing.
- The application will display processed results (object detection and tracking) in real-time or after processing, depending on the input type.

### Stopping the Application

- To stop the application, use the "Stop Processing" button at the bottom of the sidebar.
