import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
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
    return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_rgb = process_image(frame_rgb, self.model)
        return result_rgb


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
    ctx = webrtc_streamer(
        key="camera",
        video_transformer_factory=lambda: YOLOTransformer(model),
        async_transform=True,
        mode=WebRtcMode.SENDRECV,  # Use the WebRtcMode enum
        client_settings=WEBRTC_CLIENT_SETTINGS
    )

def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()

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
        process_camera(model)

if __name__ == "__main__":
    main()
