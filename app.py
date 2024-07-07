import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

@st.cache_resource
def load_model():
    return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

def process_image(image, model):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_image, _ = model.predictions(image_bgr)  # We're ignoring detected_classes here
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_rgb

def process_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_rgb = process_image(frame_rgb, model)
        stframe.image(result_rgb)
        
    vf.release()

def process_camera(model):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_rgb = process_image(frame_rgb, model)
        
        stframe.image(result_rgb)

        # Check if the camera toggle button was pressed again
        if not st.session_state.camera_on:
            break
    
    cap.release()
    st.stop()  # Stop the script cleanly when camera is turned off


def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()

    # Initialize session state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    # Input type selection
    input_type = st.radio("Select input type:", ("Image", "Video", "Camera"))
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            result_rgb = process_image(image, model)
            
            # Display the result
            st.image(result_rgb, caption='Processed Image', use_column_width=True)
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model)
            st.text("Video processing complete!")
    
    elif input_type == "Camera":
        st.write("Click the button to start or stop the camera.")
        
        # Single button to toggle camera state
        if st.button('Toggle Camera' if not st.session_state.camera_on else 'Stop Camera'):
            st.session_state.camera_on = not st.session_state.camera_on
            
            if st.session_state.camera_on:
                process_camera(model)
            else:
                st.experimental_rerun()

if __name__ == "__main__":
    main()
