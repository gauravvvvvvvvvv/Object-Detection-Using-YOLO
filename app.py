import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import logging

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

# Set up logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model():
    try:
        return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Failed to load the model. Error: {e}")
        return None

def process_image(image, model):
    try:
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR (if the image is in RGB format)
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process the image
        result_image, _ = model.predictions(image_np)
        
        # Convert back to RGB for displaying
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        return result_image_rgb
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Failed to process image. Error: {e}")
        return image_np

def process_video(video_file, model):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())

        video = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = process_image(Image.fromarray(frame_rgb), model)
            stframe.image(result, use_column_width=True)
        
        video.release()
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        st.error(f"Failed to process video. Error: {e}")

def process_camera_realtime(model):
    cap = cv2.VideoCapture(0)  # 0 for default camera
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera. Please check your camera connection.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = process_image(Image.fromarray(frame_rgb), model)
        stframe.image(result, channels="RGB", use_column_width=True)

        # Check if the user wants to stop
        if st.session_state.stop_camera:
            break

    cap.release()

def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()
    if model is None:
        return

    # Initialize session state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'stop_camera' not in st.session_state:
        st.session_state.stop_camera = False

    # Input type selection
    input_type = st.radio("Select input type:", ("Image", "Video", "Camera (Real-time)"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Process Image"):
                result = process_image(image, model)
                st.image(result, caption="Processed Image", use_column_width=True)
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            st.video(uploaded_file)
            if st.button("Process Video"):
                process_video(uploaded_file, model)
    
    elif input_type == "Camera (Real-time)":
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("Start Camera")
        with col2:
            stop_camera = st.button("Stop Camera")

        if start_camera:
            st.session_state.camera_on = True
            st.session_state.stop_camera = False
        if stop_camera:
            st.session_state.stop_camera = True
            st.session_state.camera_on = False

        if st.session_state.camera_on:
            process_camera_realtime(model)
        
        if st.session_state.stop_camera:
            st.write("Camera stopped. Click 'Start Camera' to begin again.")

if __name__ == "__main__":
    main()
