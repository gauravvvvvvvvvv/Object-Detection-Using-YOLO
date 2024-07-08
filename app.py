import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import logging

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

# Set up logging
logging.basicConfig(level=logging.INFO)

WEBRTC_CLIENT_SETTINGS = {
    "rtcConfiguration": {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    "mediaStreamConstraints": {
        "video": True,
        "audio": False,
    },
}

@st.cache_resource
def load_model():
    try:
        return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Failed to load the model. Error: {e}")
        return None

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            result_img, _ = self.model.predictions(img)
            return result_img
        except Exception as e:
            logging.error(f"Error in transform: {e}")
            return frame.to_ndarray(format="bgr24")

def process_image(image, model):
    try:
        image_np = np.array(image)
        result_image, _ = model.predictions(image_np)
        return result_image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Failed to process image. Error: {e}")
        return image_np

def process_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        try:
            result = process_image(frame, model)
            stframe.image(result, channels="BGR")
        except Exception as e:
            logging.error(f"Error processing video frame: {e}")
            break
        
    vf.release()

def process_camera(model):
    try:
        ctx = webrtc_streamer(
            key="camera",
            video_transformer_factory=lambda: YOLOTransformer(model),
            async_transform=True,
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS
        )
        
        if ctx.video_transformer:
            st.write("Camera is running. If you don't see the video, please check your browser's camera permissions.")
        else:
            st.write("Failed to start the camera. Please check your browser's camera permissions.")
    except Exception as e:
        logging.error(f"Error in process_camera: {e}")
        st.error(f"Failed to start the camera. Error: {e}")

def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()
    if model is None:
        return

    # Input type selection
    input_type = st.radio("Select input type:", ("Image", "Video", "Camera"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            result = process_image(image, model)
            
            # Display the result
            st.image(result, caption='Processed Image', use_column_width=True, channels="BGR")
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model)
            st.text("Video processing complete!")
    
    elif input_type == "Camera":
        process_camera(model)

if __name__ == "__main__":
    main()import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import logging

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

# Set up logging
logging.basicConfig(level=logging.INFO)

WEBRTC_CLIENT_SETTINGS = {
    "rtcConfiguration": {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    "mediaStreamConstraints": {
        "video": True,
        "audio": False,
    },
}

@st.cache_resource
def load_model():
    try:
        return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Failed to load the model. Error: {e}")
        return None

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            result_img, _ = self.model.predictions(img)
            return result_img
        except Exception as e:
            logging.error(f"Error in transform: {e}")
            return frame.to_ndarray(format="bgr24")

def process_image(image, model):
    try:
        image_np = np.array(image)
        result_image, _ = model.predictions(image_np)
        return result_image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Failed to process image. Error: {e}")
        return image_np

def process_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        try:
            result = process_image(frame, model)
            stframe.image(result, channels="BGR")
        except Exception as e:
            logging.error(f"Error processing video frame: {e}")
            break
        
    vf.release()

def process_camera(model):
    try:
        ctx = webrtc_streamer(
            key="camera",
            video_transformer_factory=lambda: YOLOTransformer(model),
            async_transform=True,
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS
        )
        
        if ctx.video_transformer:
            st.write("Camera is running. If you don't see the video, please check your browser's camera permissions.")
        else:
            st.write("Failed to start the camera. Please check your browser's camera permissions.")
    except Exception as e:
        logging.error(f"Error in process_camera: {e}")
        st.error(f"Failed to start the camera. Error: {e}")

def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()
    if model is None:
        return

    # Input type selection
    input_type = st.radio("Select input type:", ("Image", "Video", "Camera"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            result = process_image(image, model)
            
            # Display the result
            st.image(result, caption='Processed Image', use_column_width=True, channels="BGR")
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model)
            st.text("Video processing complete!")
    
    elif input_type == "Camera":
        process_camera(model)

if __name__ == "__main__":
    main()
