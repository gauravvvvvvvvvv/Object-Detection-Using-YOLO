import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

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
        model_path = os.path.join(os.path.dirname(__file__), 'Model', 'weights', 'best.onnx')
        data_path = os.path.join(os.path.dirname(__file__), 'data.yaml')
        return YOLO_Pred(model_path, data_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result_img, _ = self.model.predictions(img)
        return result_img

def process_image(image, model):
    try:
        image_np = np.array(image)
        result_image, _ = model.predictions(image_np)
        return result_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def process_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    stop_button = st.button("Stop Processing")
    
    try:
        while vf.isOpened() and not stop_button:
            ret, frame = vf.read()
            if not ret:
                break
            result = process_image(frame, model)
            if result is not None:
                stframe.image(result, channels="BGR")
            if st.button("Stop"):
                break
    finally:
        vf.release()
        os.unlink(tfile.name)

def process_camera(model):
    ctx = webrtc_streamer(
        key="camera",
        video_transformer_factory=lambda: YOLOTransformer(model),
        async_transform=True,
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS
    )

def main():
    st.title("Object Detection with YOLO")
    
    st.info("For best performance, use a modern browser like Chrome or Firefox.")

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
            if result is not None:
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
