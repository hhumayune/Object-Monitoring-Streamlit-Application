import streamlit as st
import os
import shutil
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import random
import time
from pytube import YouTube

def load_model(model_name):
    return YOLO("models/" + model_name)

def process_video(model, video_path, save_dir, device, tracker, use_webcam=False, youtube_url=None, break_event=None):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    elif youtube_url:
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        cap = cv2.VideoCapture(video_stream.url)
    else:
        cap = cv2.VideoCapture(video_path)

    output_path = os.path.join(save_dir, "output" + str(random.randint(1, 1000)) + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    stop_button = st.button("Break Processing")

    while cap.isOpened() and not stop_button:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, device=device, tracker=tracker + ".yaml", persist=True)  # botsort.yaml OR bytetrack.yaml
        for result in results:
            result_frame = result.plot()


            out.write(result_frame)


            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)


            frame_placeholder.image(result_frame_rgb, caption='Processed Video', channels='RGB', use_column_width=True)


        end_time = time.time()
        fps = 1 / (end_time - start_time)
        height = frame.shape[0]
        width = frame.shape[1]

        info_placeholder.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="text-align: center;">
                <h3>Height</h3>
                <p>{height}</p>
            </div>
            <div style="text-align: center;">
                <h3>Width</h3>
                <p>{width}</p>
            </div>
            <div style="text-align: center;">
                <h3>FPS</h3>
                <p>{fps:.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    cap.release()
    out.release()

    return output_path



def process_image(model, image_path, save_dir, device, tracker):
    results = model.track(image_path, device=device, save=True, project=save_dir, tracker=tracker + ".yaml")
    return results


def main():
    st.title("YOLOv8 Object Detection and Tracking")


    if 'processing' not in st.session_state:
        st.session_state.processing = False
        st.session_state.processed_video_path = None

    st.sidebar.header("DL Model Config")
    input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video", "Webcam", "YouTube"])
    model_name = st.sidebar.selectbox("Select Model", ["yolov8n.pt", "yolov8m.pt"])

    hardware = st.sidebar.selectbox("Select Hardware", ["PC", "GPU (CUDA)"])
    if hardware == "GPU (CUDA)":
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            st.sidebar.write("No GPU (CUDA) Found. Selecting CPU")
            device = 'cpu'
    else:
        device = 'cpu'

    tracker = st.sidebar.selectbox("Select tracker", ["bytetrack", "botsort"])
    confidence_threshold = st.sidebar.slider("Select Model Confidence", 30, 100, 50) / 100

    file = None
    youtube_url = None
    if input_type == "Image":
        file = st.sidebar.file_uploader("Choose an Image...", type=["jpeg", "png"])
    elif input_type == "Video":
        file = st.sidebar.file_uploader("Choose a Video...", type=["mp4", "avi"])
    elif input_type == "YouTube":
        youtube_url = st.sidebar.text_input("Enter YouTube Video URL")

    if file is not None or input_type == "Webcam" or (input_type == "YouTube" and youtube_url):

        if not os.path.exists("imagesVideos"):
            os.makedirs("imagesVideos")

        file_path = None
        if file is not None:

            file_path = os.path.join("imagesVideos", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())


            print(f"File path: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")


        model = load_model(model_name)

        model.conf = confidence_threshold

        save_dir = "./result"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        if input_type == "Webcam":
            st.info("Processing webcam input...")
            if st.session_state.processing:
                st.session_state.processing = False
            result_video_path = process_video(model, file_path, save_dir, device, tracker, use_webcam=True)
            st.session_state.processed_video_path = result_video_path
            st.success("Object detection and tracking from webcam completed!")

        elif input_type == "YouTube" and youtube_url:
            st.info("Processing YouTube video input...")
            if st.session_state.processing:
                st.session_state.processing = False
            result_video_path = process_video(model, file_path, save_dir, device, tracker, youtube_url=youtube_url)
            st.session_state.processed_video_path = result_video_path
            st.success("Object detection and tracking from YouTube video completed!")

        elif file_path and file_path.endswith((".dng", ".png", ".webp", ".mpo", ".bmp", ".tiff", ".pfm", ".jpg", ".tif", ".jpeg")):

            results = process_image(model, file_path, save_dir, device, tracker)
            st.success("Object detection completed!")
            result_image_path = os.path.join(save_dir, "track", os.listdir(os.path.join(save_dir, "track"))[0])
            result_image = Image.open(result_image_path)
            st.image(result_image, caption='Processed Image')

            with open(result_image_path, "rb") as result_file:
                st.download_button(
                    label="Download Result Image",
                    data=result_file,
                    file_name="result_image" + os.path.splitext(file_path)[1],
                    mime="image/" + os.path.splitext(file_path)[1][1:]
                )

        elif file_path and file_path.endswith((".gif", ".mov", ".ts", ".webm", ".mkv", ".wmv", ".mpeg", ".avi", ".m4v", ".mp4", ".asf", ".mpg")):

            st.session_state.processing = True
            result_video_path = process_video(model, file_path, save_dir, device, tracker)
            st.session_state.processed_video_path = result_video_path
            st.session_state.processing = False
            st.success("Object detection and tracking completed!")

            with open(result_video_path, "rb") as result_file:
                st.download_button(
                    label="Download Result Video",
                    data=result_file,
                    file_name="result_video.mp4",
                    mime="video/mp4"
                )

    else:
        st.info("Please upload an image or video file, select webcam, or enter a YouTube URL to proceed.")

    if st.session_state.processing:
        break_event = st.button("Break Processing")
        if break_event:
            st.session_state.processing = False
            st.error("Processing interrupted.")


if __name__ == "__main__":
    main()
